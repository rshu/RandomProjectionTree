import BuildRPTree as RP


class TreeNode:
    def __init__(self, value=[]):
        self.key = value
        self.left = None
        self.right = None

    # return the object representation
    def __repr__(self):
        left = None if self.left is None else self.left.key
        right = None if self.right is None else self.right.key
        return '(Current Node: {}, Left: {}, Right: {})'.format(self.key, left, right)


# def main():
#     # root = TreeNode([2000, 2001])
#     # root.left = TreeNode([1005, 1006, 1007])
#     # root.right = TreeNode([1008, 1009, 1010])
#     # print(repr(root))
#
#     print("Building Random Projection Tree...")

# if __name__ == "__main__":
#     main()
