import torch

from src.attacks.registry import register

device = "cuda" if torch.cuda.is_available() else "cpu"


@register
class FGSM:
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT : 0.007)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,
        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, args)
        >>> adv_images = attack(images, labels, eps)

    """

    def __init__(self, criterion, args):
        # super(FGSM, self).__init__("FGSM", model)
        self.args = args
        self.eps = 0.007
        self.criterion = criterion

    def forward(self, model, images, labels, eps=None):
        r"""
        Overridden.
        """
        images = images.to(device)
        labels = labels.to(device)
        # labels = self._transform_label(images, labels)
        if eps is None:
            eps = self.eps

        images.requires_grad = True
        outputs = model(images)
        cost = self.criterion(outputs, labels).to(device)

        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        adv_images = images + eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
