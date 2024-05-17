from __future__ import annotations

import filecmp
from pathlib import Path

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"
TEST_TIME = "2024-04-04 00:00"


class dircmp(filecmp.dircmp):
    """add test utils to the original filecmp.dircmp"""

    def is_same(self) -> bool:
        """returns True if dir content is the same"""
        if self.left_only or self.right_only or self.diff_files or self.funny_files:
            return False
        return True

    def msg(self) -> str:
        """
        returns a report of the differences between left and right.
        It's the same as .report() but, instead of printing it to stdout,
        it returns it as a string
        """

        msg = f"diff {self.left} {self.right}\n"
        if self.left_only:
            self.left_only.sort()
            msg += f"Only in {self.left} : {self.left_only}"
        if self.right_only:
            self.right_only.sort()
            msg += f"Only in {self.right} : {self.right_only}"
        if self.same_files:
            self.same_files.sort()
            msg += f"Identical files : {self.same_files}"
        if self.diff_files:
            self.diff_files.sort()
            msg += f"Differing files : {self.diff_files}"
        if self.funny_files:
            self.funny_files.sort()
            msg += f"Trouble with common files : {self.funny_files}"
        if self.common_dirs:
            self.common_dirs.sort()
            msg += f"Common subdirectories : {self.common_dirs}"
        if self.common_funny:
            self.common_funny.sort()
            msg += f"Common funny cases : {self.common_funny}"

        return msg
