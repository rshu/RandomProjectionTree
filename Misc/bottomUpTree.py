class Node:
    def __init__(self, value):
        self.key = value
        self.left = None
        self.right = None

    # return the object representation
    def __repr__(self):
        left = None if self.left is None else self.left.key
        right = None if self.right is None else self.right.key
        return '(Key: {}, Left: {}, Right: {})'.format(self.key, left, right)

def maxLeafNode(root):
    if root is None:
        return

    if root.left is not None or root.right is not None:
        maxLeafNode(root.left)
        maxLeafNode(root.right)
        root.key = max(root.left.key, root.right.key) # update this based on fitness function

    return root


def main():

    print("Build a tree first...")
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

    print(repr(root))

    maxLeafNode(root)
    print("After pruning:")
    print(root.key)
    print(repr(root.left.left.left))


if __name__ == "__main__":
    main()
