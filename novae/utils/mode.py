class Mode:
    """Novae mode class, used to store states variables related to training and inference."""

    def __init__(
        self,
        zero_shot: bool = False,
        trained: bool = False,
        pretrained: bool = False,
        multimodal: bool = False,
    ):
        self.zero_shot = zero_shot
        self.trained = trained
        self.pretrained = pretrained
        self.multimodal = multimodal

    def __repr__(self) -> str:
        return f"Mode({dict(self.__dict__.items())})"

    ### Mode modifiers

    def from_pretrained(self):
        self.zero_shot = False
        self.trained = True
        self.pretrained = True

    def fine_tune(self):
        assert self.pretrained, "Fine-tuning requires a pretrained model."
        self.zero_shot = False

    def fit(self):
        self.zero_shot = False
        self.trained = False
