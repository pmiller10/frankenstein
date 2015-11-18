class MissingConfig(Warning):
    msg = """
          Warning: No config file defined.
          In client app make a config.py file.
          Example:
          from FML.default_config import DefaultConfig
          class Config(DefaultConfig):
              @classmethod
              def loss(cls, preds, targets):
                  return sum([(p-t)**2 for p,t in zip(preds, targets)])
          """


try:
    from config import Config
except ImportError:
    print MissingConfig(MissingConfig.msg)
    # if config.py is not defined, then use default_config.py
    from default_config import DefaultConfig as Config
