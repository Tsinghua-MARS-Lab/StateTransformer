class A:
    def __init__(self, config):
        print("A.__init__")
        print(config)
        super().__init__()

    def print(self):
        print("A.print")

class B:
    def __init__(self):
        print("B.__init__")
        super().__init__()

    def print(self):
        print("B.print")

class C(A, B):
    def __init__(self, config):
        print("C.__init__")
        super().__init__(config)

c = C(config="config")
print(C.__mro__)
c.print()