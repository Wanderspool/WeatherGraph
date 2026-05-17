class Runner:
    class Graphs:
        def __init__(self):
            self.nodes = {}
        def jraph(self):
            return self

    def __init__(self, verbose=False, config=None):
        self.config = config
        self.static_graphs = {
            "e": self.Graphs(),
            "p": self.Graphs(),
            "d": self.Graphs()
        }
        class Transformed:
            def apply(self, params, graphs, i_time):
                return graphs
        self.transformed = Transformed()

    def init_set(self, data):
        return data
