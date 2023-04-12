import pytest
class NotInRange(Exception):
    def __init__(self):
        self.message = "value not in range"
        super().__init__(self.message)

def test_generic():
    a =1
    with pytest.raises(NotInRange):
        if a not in range(10,20):
            raise NotInRange
        
def test_something():
    a =2
    b =3
    assert True
    