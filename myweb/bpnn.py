#参考博客：https://www.cnblogs.com/hhh5460/p/4304628.html
import math
import random

random.seed(0)
def rand(a, b):
    return (b - a) * random.random() + a

def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []

    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]#输入层输出值
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]#隐藏层输入值
            self.hidden_cells[j] = sigmoid(total)#隐藏层的输出值
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            # self.output_cells[k] = sigmoid(total)
            self.output_cells[k] =total#输出层的激励函数是f(x)=x
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):#x,y,修改最大迭代次数， 学习率λλ， 矫正率μμ三个参数.
        # feed forward
        self.predict(case)
        # get output layer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            # output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
            output_deltas[o] = error

        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error

        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]#？？？？？？？？？？
                self.output_correction[h][o] = change

        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)

    def test(self,cases,labels,ni, nh, no):
        self.setup(ni, nh, no)
        self.train(cases, labels, 10000, 0.05, 0.1)
        res = []
        for case in cases:
            res.extend(self.predict(case))
        return res

if __name__ == '__main__':
    # cases=[ [564, 565, 567, 568], [551, 552, 553, 554], [537, 539, 540, 541],
    #         [525, 526, 527, 528], [512, 513, 514, 515], [500, 501, 502, 503],
    #         [488, 489, 490, 491]]
    # labels=[[571], [551], [527], [533], [510], [512], [484]]
    # cases = [[10, 2],[4, 7],[1, 0],[8, 1],]
    # labels = [[3], [9], [1], [2]]
    cases=[[0.0313,0.313*2,0.938*3,0.4688*4,0.375*5],]
    nn = BPNeuralNetwork()
    res=nn.test(cases,labels,2,6,1)
    print(res)
