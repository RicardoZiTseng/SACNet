from . import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

print("\n\nPlease cite the following paper when using SACNet. Oh shit, we don't have paper yet.")
print("If you have any questions, please contact Dr.Zeng (zilongzeng@mail.bnu.edu.cn) or Dr.Zhao (tengdazhao@bnu.edu.cn).\n")

try:
    from art import *
    tprint("SACNet")
except:
    pass
