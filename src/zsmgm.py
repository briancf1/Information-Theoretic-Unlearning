import torch
import torch.nn.functional as F

from datasets import CIFAR_MEAN, CIFAR_STD


def _repeat_batch(tensor, repeats):
    return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))


def _normalized_input_bounds():
    mins = [(0.0 - mean) / std for mean, std in zip(CIFAR_MEAN, CIFAR_STD)]
    maxs = [(1.0 - mean) / std for mean, std in zip(CIFAR_MEAN, CIFAR_STD)]
    return min(mins), max(maxs)


def _unwrap_logits(output):
    if isinstance(output, tuple):
        return output[0]
    return output


class ZSMGM:
    def __init__(self, model, optimizer, device, parameters):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.epsilon = parameters["epsilon"]
        self.k_neighbors = parameters["k_neighbors"]
        self.pgd_steps = parameters["pgd_steps"]
        self.pgd_alpha = parameters["pgd_alpha"]
        self.lambda_manifold = parameters["lambda_manifold"]

        clip_min, clip_max = _normalized_input_bounds()
        self.clip_min = parameters.get("clip_min", clip_min)
        self.clip_max = parameters.get("clip_max", clip_max)

        self._latent_cache = {}
        self._hook_handle = self._register_latent_hook()

    def _register_latent_hook(self):
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "classifier"):
            raise ValueError("ZS-MGM currently expects the Pins VGG16 wrapper.")

        try:
            return self.model.model.classifier[3].register_forward_hook(self._capture_latent)
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("ZS-MGM could not locate the VGG penultimate layer.") from exc

    def _capture_latent(self, _module, _inputs, output):
        self._latent_cache["latent"] = output

    def _forward_with_latents(self, inputs):
        self._latent_cache.pop("latent", None)
        logits = _unwrap_logits(self.model(inputs))
        latent = self._latent_cache.get("latent")

        if latent is None:
            raise RuntimeError("ZS-MGM forward hook did not capture the penultimate activations.")

        return logits, latent

    def close(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def modify_weight(self, forget_loader):
        criterion = torch.nn.CrossEntropyLoss()
        proxy_criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        final_loss = None

        for inputs, _labels, class_labels in forget_loader:
            inputs = inputs.to(self.device)
            class_labels = class_labels.to(self.device)

            with torch.no_grad():
                _, h_orig = self._forward_with_latents(inputs)
                h_orig = h_orig.detach()

            batch_size = int(inputs.size(0))
            expanded_x = _repeat_batch(inputs, self.k_neighbors)
            expanded_h_orig = _repeat_batch(h_orig, self.k_neighbors)

            noise = torch.randn_like(expanded_x)
            flat_noise = noise.reshape(noise.size(0), -1)
            noise_norm = flat_noise.norm(p=2, dim=1).clamp_min(1e-12)
            radius = torch.rand(
                noise.size(0), device=noise.device, dtype=noise.dtype
            ) * self.epsilon
            scale = (radius / noise_norm).view(-1, *([1] * (noise.ndim - 1)))
            zeta = (noise * scale).detach().requires_grad_(True)

            for _ in range(self.pgd_steps):
                x_adv = torch.clamp(
                    expanded_x + zeta,
                    min=self.clip_min,
                    max=self.clip_max,
                )
                logits_adv, h_adv = self._forward_with_latents(x_adv)

                logits_grouped = logits_adv.reshape(self.k_neighbors, batch_size, -1)
                probs = F.softmax(logits_grouped, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean(dim=1)

                h_adv_grouped = h_adv.reshape(self.k_neighbors, batch_size, -1)
                h_orig_grouped = expanded_h_orig.reshape(self.k_neighbors, batch_size, -1)
                latent_penalty = (
                    self.lambda_manifold
                    * F.mse_loss(
                        h_adv_grouped,
                        h_orig_grouped,
                        reduction="none",
                    ).mean(dim=(1, 2))
                    * h_orig.size(-1)
                )

                adv_loss = (entropy - latent_penalty).sum()
                (grad_zeta,) = torch.autograd.grad(
                    adv_loss,
                    zeta,
                    retain_graph=False,
                    create_graph=False,
                )

                with torch.no_grad():
                    flat_grad = grad_zeta.reshape(grad_zeta.size(0), -1)
                    grad_norm = flat_grad.norm(p=2, dim=1).clamp_min(1e-12)
                    unit_grad = (flat_grad / grad_norm.unsqueeze(1)).reshape_as(grad_zeta)
                    zeta = zeta + self.pgd_alpha * unit_grad

                    flat_zeta = zeta.reshape(zeta.size(0), -1)
                    zeta_norm = flat_zeta.norm(p=2, dim=1)
                    factor = torch.clamp(
                        self.epsilon / zeta_norm.clamp_min(1e-12),
                        max=1.0,
                    ).view(-1, *([1] * (zeta.ndim - 1)))
                    zeta = zeta * factor

                zeta = zeta.detach().requires_grad_(True)

            with torch.no_grad():
                proxies = torch.clamp(
                    expanded_x + zeta.detach(),
                    min=self.clip_min,
                    max=self.clip_max,
                )

            self.optimizer.zero_grad()

            logits_forget = _unwrap_logits(self.model(inputs))
            loss_forget = -criterion(logits_forget, class_labels)
            loss_forget.backward()

            logits_proxy = _unwrap_logits(self.model(proxies))
            pseudo_labels = logits_proxy.detach().argmax(dim=-1)
            loss_proxy = proxy_criterion(logits_proxy, pseudo_labels) / (
                self.k_neighbors * batch_size
            )
            loss_proxy.backward()

            self.optimizer.step()
            final_loss = float(loss_forget.item() + loss_proxy.item())

        return final_loss