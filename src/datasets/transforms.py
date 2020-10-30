class CustomTransform(object):
    def __call__(self, input):
        return input['img']

    def __repr__(self):
        return self.__class__.__name__