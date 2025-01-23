class Vertex:
    def __init__(self, v, id) -> None:
        self.id = id
        self.val = v
        self.adjacent = list()

    def addEdge(self, id):
        if id in self.adjacent:
            raise "dup edge added"
        else:
            self.adjacent.append(id)

    def delEdge(self, id):
        self.adjacent.remove(id)


class Graph:
    def __init__(self) -> None:
        self.color = None
        self.vertexs = list()

    def addVertex(self, v):
        id = len(self.vertexs)
        self.vertexs.append(Vertex(v, id))
        return id

    def addEdge(self, a1, n1, a2, n2):
        id1 = 0
        while id1 < len(self.vertexs):
            if self.vertexs[id1].val == (a1, n1):
                break
            id1 += 1
        if id1 == len(self.vertexs):
            id1 = self.addVertex((a1, n1))

        id2 = 0
        while id2 < len(self.vertexs):
            if self.vertexs[id2].val == (a2, n2):
                break
            id2 += 1
        if id2 == len(self.vertexs):
            id2 = self.addVertex((a2, n2))

        self.vertexs[id1].addEdge(id2)

    def deleteEdge(self, a1, n1, a2, n2):
        id1 = 0
        while id1 < len(self.vertexs):
            if self.vertexs[id1].val == (a1, n1):
                break
            id1 += 1

        id2 = 0
        while id2 < len(self.vertexs):
            if self.vertexs[id2].val == (a2, n2):
                break
            id2 += 1

        self.vertexs[id1].delEdge(id2)

    def cycleDetect(self) -> bool:
        self.color = [0] * len(self.vertexs)  # 0 white, 1 gray, 2 black
        for no, color in enumerate(self.color):
            if color == 0 and self.dfs(no):
                return True
        return False

    def dfs(self, start) -> bool:
        self.color[start] = 1
        for id in self.vertexs[start].adjacent:
            if self.color[id] == 0:
                if self.dfs(id):
                    return True
            elif self.color[id] == 1:
                return True
        self.color[start] = 2
        return False
