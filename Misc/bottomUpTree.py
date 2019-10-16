# compare two children of a node
# if left > right, replace parent with left, otherwise with right
# or replace parent with sum of left + right
# bottom up to root, and print root


class treeNode:
    def __init__(self, index):
        self.key = index
        self.left = None
        self.right = None
        self.parent = None


# print all nodes that do not have sibling

def printNoSiblingNode(root):
    if root is None:
        return

    if root.left is not None and root.right is not None:
        printNoSiblingNode(root.left)
        printNoSiblingNode(root.right)
    elif root.left is not None:
        print(root.left.key)
        printNoSiblingNode(root.left)
    elif root.right is not None:
        print(root.right.key)
        printNoSiblingNode(root.right)


# Algorithm Inorder(tree)
#    1. Traverse the left subtree, i.e., call Inorder(left-subtree)
#    2. Visit the root.
#    3. Traverse the right subtree, i.e., call Inorder(right-subtree)
def printInOrder(root):
    if root:
        printInOrder(root.left)
        print(root.key)
        printInOrder(root.right)


# Algorithm Postorder(tree)
#    1. Traverse the left subtree, i.e., call Postorder(left-subtree)
#    2. Traverse the right subtree, i.e., call Postorder(right-subtree)
#    3. Visit the root.
def printPostOrder(root):
    if root:
        printPostOrder(root.left)
        printPostOrder(root.right)
        print(root.key)


# Algorithm Preorder(tree)
#    1. Visit the root.
#    2. Traverse the left subtree, i.e., call Preorder(left-subtree)
#    3. Traverse the right subtree, i.e., call Preorder(right-subtree)
def printPreOrder(root):
    if root:
        print(root.key)
        printPreOrder(root.left)
        printPreOrder(root.right)


# def insert(node, key):
#     if node == None:
#         return treeNode(key)
#
#     if key < node.key:
#         left = insert(node.left, key)
#         node.left = left
#         left.parent = node
#     elif key > node.key:
#         right = insert(node.right, key)
#         node.right = right
#         right.parent = node
#     return node


def main():
    root = treeNode(1)
    root.left = treeNode(2)
    root.right = treeNode(3)
    root.left.right = treeNode(4)
    root.right.left = treeNode(5)
    root.right.left.right = treeNode(6)

    print("\nNode with no siblings")
    printNoSiblingNode(root)

    print("\nPreorder traversal of binary tree is")
    printPreOrder(root)

    print("\nInorder traversal of binary tree is")
    printInOrder(root)

    print("\nPostorder traversal of binary tree is")
    printPostOrder(root)


if __name__ == '__main__':
    main()
