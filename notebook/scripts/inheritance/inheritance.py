"""
How to call GrandParent class method from Child class
"""
class GrandParent:
    @classmethod
    def run(cls):
        print("Call GrandParent")
        print(cls.__name__)

class Parent(GrandParent):
    @classmethod
    def run(cls):
        print("Call Parent")
        print(cls.__name__)

class Child(Parent):
    @classmethod
    def run(cls):
        print("Call Child")
        print(cls.__name__)

    @classmethod
    def run_parent(cls):
        super().run()

    @classmethod
    def run_grandparent(cls):
        super(Parent, cls).run()

if __name__ == "__main__":
    Child.run_parent()
    Child.run_grandparent()