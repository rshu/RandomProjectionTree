class Node:
    def __init__(self, value):
        self.key = value
        self.left = None
        self.right = None


def maxLeafNode(root):
    if root is None:
        return

    if root.left is not None or root.right is not None:
        maxLeafNode(root.left)
        maxLeafNode(root.right)
        root.key = max(root.left.key, root.right.key)

    return root


def main():
    print("Build a tree first")
    root = Node(2000)
    root.left = Node(1005)
    root.right = Node(923)
    root.left.left = Node(465)
    root.left.right = Node(503)
    root.right.left = Node(300)
    root.right.right = Node(600)
    root.left.left.left = Node(230)
    root.left.left.right = Node(115)
    root.left.right.left = Node(10)
    root.left.right.right = Node(423)
    root.right.left.left = Node(100)
    root.right.left.right = Node(230)
    root.right.right.left = Node(378)
    root.right.right.right = Node(223)

    maxLeafNode(root)
    print(root.key)


if __name__ == "__main__":
    main()
