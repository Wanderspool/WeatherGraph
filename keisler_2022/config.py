class Config:
    class Data:
        weights_file = "dummy_weights.pkl"
    def __init__(self):
        self.data = self.Data()
    def resolve_artifact(self, path):
        return path
