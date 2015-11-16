import random



class IntGen():

    def __init__(self, min, max):
        self.min = min
        self.max = max


    def generate(self):
        return random.randint(self.min, self.max)



class FloatGen():

    def __init__(self, min, max):
        self.min = min
        self.max = max


    def generate(self):
        return round(random.uniform(self.min, self.max), 3)



class StrGen():

    def __init__(self, options):
        self.options = options


    def generate(self):
        return random.choice(self.options)



# TODO add limits from config.py
# TODO add another level for model-specific limits
param_limits = {
                'C': IntGen(1, 10),
                'tol': FloatGen(0.0001, 1.),
                'penalty': StrGen(['l1', 'l2']),
                'degree': IntGen(1, 5)
               }

def generate(klass):
    random_params = {}
    for key in klass().get_params():
        if key in param_limits:
            random_params[key] = param_limits[key].generate()
    return klass(**random_params).get_params()
