# Python3 program to demonstrate insert operation
# in binary search tree with parent pointer


# (a) Inorder (Left, Root, Right)
# (b) Preorder (Root, Left, Right)
# (c) Postorder (Left, Right, Root)


# A utility function to create a new BST Node
class treeNode:
    def __init__(self, index):
        self.key = index
        self.left = None
        self.right = None
        self.parent = None


# A utility function to do inorder
# traversal of BST
def inorder(root):
    if root != None:
        inorder(root.left)
        print("Node :", root.key, ", ", end="")
        if root.parent == None:
            print("Parent : NULL")
        else:
            print("Parent : ", root.parent.key)
        inorder(root.right)

    # A utility function to insert a new


# Node with given key in BST
def insert(node, key):
    # If the tree is empty, return a new Node
    if node == None:
        return treeNode(key)

        # Otherwise, recur down the tree
    if key < node.key:
        lchild = insert(node.left, key)
        node.left = lchild

        # Set parent of root of left subtree
        lchild.parent = node
    elif key > node.key:
        rchild = insert(node.right, key)
        node.right = rchild

        # Set parent of root of right subtree
        rchild.parent = node

        # return the (unchanged) Node pointer
    return node


# Driver Code
if __name__ == '__main__':
    # Let us create following BST
    #         50
    #     /     \
    #     30     70
    #     / \ / \
    # 20 40 60 80
    root = None
    root = insert(root, 50)
    insert(root, 30)
    insert(root, 20)
    insert(root, 40)
    insert(root, 70)
    insert(root, 60)
    insert(root, 80)

    # print iNoder traversal of the BST
    inorder(root)