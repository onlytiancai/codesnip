from eltdx import F10Client

f10 = F10Client(timeout=3)
print(f10.company_profile("000034").rows[0])
