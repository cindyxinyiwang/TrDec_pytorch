import six


class HParams(object):
  def __init__(self, **kwargs):
    #super(Iwslt16EnDeBpe32SharedParams, self).__init__(**kwargs)
    for name, value in six.iteritems(kwargs):
      self.add_param(name, value)

    self.dataset = "Hparams"

    self.unk = "<unk>"
    self.bos = "<s>"
    self.eos = "</s>"
    self.pad = "<pad>"

    self.unk_id = None
    self.eos_id = None
    self.bos_id = None
    self.pad_id = None

    self.tiny = 0.
    self.inf = float("inf")


  def add_param(self, name, value):
    setattr(self, name, value)
