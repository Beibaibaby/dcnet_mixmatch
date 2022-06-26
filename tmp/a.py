class Parent1():
    def __init__(self, a):
        self.a = a


class Parent2():
    def __init__(self, b):
        self.b = b


class Child(Parent1, Parent2):
    def __init__(self, a, b):
        super().__init__()
