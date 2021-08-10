from utils import *

class Node():
    def __init__(self, depth=0, parent=None):
        self.visits = 1.0
        self.success = 0.0
        self.rchild = None
        self.lchild = None
        self.mchild = None
        self.parent = parent
        # self.tag = tag
        # self.psb = None
        # self.sublattice_size = len(state)

        if parent is None:
            self.depth = depth
        else:
            self.depth = self.parent.depth + 1

    def calculate_score(self, node):
        term1 = node.success / node.visits
        term2 = 1.0 / math.sqrt(2.0) * math.sqrt(math.log(node.parent.visits) / node.visits)
        return term1 + term2

    def ana_score(self, node):
        term1 = node.success / node.visits
        term2 = 1.0 / math.sqrt(2.0) * math.sqrt(math.log(node.parent.visits) / node.visits)
        return term1, term2

    def best_child(self):
        sc = []
        depth = self.depth
        x_idx = int(get_idx()[depth] / 128)
        y_idx = int(get_idx()[depth] % 128)
        if self.rchild is not None:
            sc.append(self.calculate_score(self.rchild))
        else:
            sc.append(-9999)
        if self.lchild is not None:
            sc.append(self.calculate_score(self.lchild))
        else:
            sc.append(-9999)
        if self.mchild is not None:
            sc.append(self.calculate_score(self.mchild))
        else:
            sc.append(-9999)

        max_score = max(sc)

        if max_score == sc[0]:
            set_state_element(x_idx, y_idx, 1)
            return self.rchild
        elif max_score == sc[1]:
            set_state_element(x_idx, y_idx, -1)
            return self.lchild
        else:
            set_state_element(x_idx, y_idx, 0)
            return self.mchild


    def UCTSearch(self):
        lnode = self
        while not lnode.is_leaf():
            if lnode.lchild is not None and lnode.rchild is not None and lnode.mchild is not None:
                lnode = lnode.best_child()
            else:
                lnode = lnode.expand()
                break
        return lnode

        # if self.is_leaf():
        #     return self
        # else:
        #     if self.lchild is not None and self.rchild is not None and self.mchild is not None:
        #         b_child = self.best_child()
        #         lnode = b_child.UCTSearch()
        #     else:
        #         lnode = self.expand()
        #     return lnode


    def expand(self):
        tmp = [0.5,0.5]
        new_depth = self.depth + 1
        new_x_idx = int(get_idx()[new_depth] / 128)
        new_y_idx = int(get_idx()[new_depth] % 128)
        new_node = Node(parent=self)
        if self.lchild is None and self.rchild is None and self.mchild is None:
            dire = random.randint(-1, 1)
            if dire is 1:
                self.rchild = new_node
            elif dire is 0:
                self.mchild = new_node
            else:
                self.lchild = new_node
        elif self.lchild is None and self.rchild is None:
                dire = np.random.choice([-1, 1], p=tmp)
                if dire is -1:
                    self.lchild = new_node
                else:
                    self.rchild = new_node
        elif self.lchild is None and self.mchild is None:
                dire = np.random.choice([-1, 0], p=tmp)
                if dire is -1:
                    self.lchild = new_node
                else:
                    self.mchild = new_node
        elif self.mchild is None and self.rchild is None:
                dire = np.random.choice([0, 1], p=tmp)
                if dire is 0:
                    self.mchild = new_node
                else:
                    self.rchild = new_node
        else:
            if self.rchild is None:
                dire = 1
                self.rchild = new_node
            elif self.mchild is None:
                dire=0
                self.mchild = new_node
            else:
                dire=-1
                self.lchild = new_node
        set_state_element(new_x_idx, new_y_idx, dire)
        return new_node


    # def UCTSearch(self):  ##return leaf_node
    #
    #     ###CNN strategy and UCTscore
    #
    #     if self.is_leaf():
    #         return self
    #     else:
    #         if self.lchild is None and self.rchild is None and self.mchild is None:
    #             lnode = self.expand()
    #         elif self.lchild is not None and self.rchild is not None and self.mchild is not None:
    #             b_child = self.best_child()
    #             lnode = b_child.UCTSearch()
    #         else:
    #             tmp = []
    #             i = self.depth // self.sublattice_size
    #             j = self.depth % self.sublattice_size
    #             new_state = self.state
    #             tmp.append(0.5)
    #             tmp.append(0.5)
    #             if self.lchild is None and self.rchild is None:
    #
    #                 dire = np.random.choice([-1, 1], p=tmp)
    #                 new_state[i][j] = dire
    #                 if dire is -1:
    #                     self.lchild = Node(new_state, parent=self, tag=-1)
    #                     lnode = self.lchild.expand()
    #                 else:
    #                     self.rchild = Node(new_state, parent=self, tag=1)
    #                     lnode = self.rchild.expand()
    #             elif self.lchild is None and self.mchild is None:
    #
    #                 dire = np.random.choice([-1, 0], p=tmp)
    #                 new_state[i][j] = dire
    #                 if dire is -1:
    #                     self.lchild = Node(new_state, parent=self, tag=-1)
    #                     lnode = self.lchild.expand()
    #                 else:
    #                     self.mchild = Node(new_state, parent=self, tag=0)
    #                     lnode = self.mchild.expand()
    #             elif self.mchild is None and self.rchild is None:
    #
    #                 tmp = normalization(tmp)
    #
    #                 dire = np.random.choice([0, 1], p=tmp)
    #                 new_state[i][j] = dire
    #                 if dire is 0:
    #                     self.mchild = Node(new_state, parent=self, tag=0)
    #                     lnode = self.mchild.expand()
    #                 else:
    #                     self.rchild = Node(new_state, parent=self, tag=1)
    #                     lnode = self.rchild.expand()
    #             else:
    #                 if self.rchild is None:
    #                     new_state[i][j] = 1
    #                     self.rchild = Node(new_state, parent=self, tag=1)
    #                     lnode = self.rchild.expand()
    #                 elif self.mchild is None:
    #                     new_state[i][j] = 0
    #                     self.mchild = Node(new_state, parent=self, tag=0)
    #                     lnode = self.mchild.expand()
    #                 else:
    #                     new_state[i][j] = -1
    #                     self.lchild = Node(new_state, parent=self, tag=-1)
    #                     lnode = self.lchild.expand()
    #
    #         return lnode

    def is_leaf(self):
        return self.depth == get_state().size*0.5

    # def expand(self):
    #     if self.is_leaf():
    #         return self
    #     else:
    #         dire = random.randint(-1, 1)
    #         i = self.depth // self.sublattice_size
    #         j = self.depth % self.sublattice_size
    #         new_state = self.state
    #         if dire is 1:
    #             new_state[i][j] = 1
    #             self.rchild = Node(new_state, parent=self, tag=1)
    #             l_node = self.rchild.expand()
    #         elif dire is 0:
    #             new_state[i][j] = 0
    #             self.mchild = Node(new_state, parent=self, tag=0)
    #             l_node = self.mchild.expand()
    #         else:
    #             new_state[i][j] = -1
    #             self.lchild = Node(new_state, parent=self, tag=-1)
    #             l_node = self.lchild.expand()
    #         return l_node


    def backpropagation(self, val):
        parent = self.parent
        self.success += val
        self.visits += 1
        if parent is None:
            return self
        self.parent.backpropagation(val)

    # def is_rc(self):
    #     if self.tag is 1:
    #         return True

    # def get_direction(self):
    #     bc = self.best_child()
    #     return bc.tag





