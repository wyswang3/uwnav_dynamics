from __future__ import annotations

import sys
from pathlib import Path


# 仓库采用 src layout，本地跑 pytest 时不要求先安装成 site-packages。
# 这里只给测试会话补充导入路径，不影响运行时代码与打包行为。
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
