from abc import ABC


class Controller(ABC):
    # Sketch of what a generic controller could look like
    @staticmethod
    def __call__(params, controller_state, system_state, t, covariates=None):
        pass

    @staticmethod
    def update(params, controller_state, system_state, t, covariates=None):
        pass

    @staticmethod
    def init(params):
        # returns initial controller state
        pass
