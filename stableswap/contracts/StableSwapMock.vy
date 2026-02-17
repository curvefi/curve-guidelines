# pragma version 0.4.3

A: public(uint256)
price_oracle: public(uint256)

@external
def set(A: uint256, p: uint256):
    self.A = A
    self.price_oracle = p
