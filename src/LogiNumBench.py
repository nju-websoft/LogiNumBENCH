import random
import copy
from time import time
import myGraph
from queue import Queue


class RandomGene:
    """
    manage the pools and wrap the randomness
    """
    namePool = ['Alice', 'Bob', 'Charlie', 'Dave', 'Erin', 'Fiona', 'Gary', 'Harry', 'Amelia', 'Sophia', 'Olivia',
                'Emma', 'James', 'William', 'Michael', 'Benjamin', 'Emily', 'Grace', 'Hannah', 'Samuel', 'Daniel',
                'Joseph', 'Alexander', 'Chioe', 'Mia', 'Liam']
    attrPool = ['red', 'blue', 'green', 'kind', 'nice', 'big', 'cold', 'young', 'round', 'rough', 'white', 'smart',
                'quiet', 'furry', 'confident', 'friendly', 'compassionate', 'considerate', 'generous', 'gruff',
                'reserved']
    relationPool = ['likes', 'chases', 'eats', 'sees', 'visits', 'needs', 'hugs', 'greets', 'helps', 'argues', 'kisses',
                    'meets', 'interacts']
    fact1Pool = ['{e}\'s {a} is {num}']
    fact2Pool = ['{ei} {r} {ej}']
    implictionPool = ['if {condition}, then {conclusion}']
    exprPool = ['{y_a} multiplied by {k} and then adding {m}']
    expr1Pool = ['{y_a} plus {m}']
    y_aPool = ['y\'s {a}']

    @classmethod
    def set_pools(cls, diction):
        """
        diction: a dictionary with keys 'namePool', 'attrPool', 'relationPool'
        sets the pools of names, attributes, and relations
        """
        cls.namePool = diction['namePool']
        cls.attrPool = diction['attrPool']
        cls.relationPool = diction['relationPool']

    @classmethod
    def load_config(cls, diction):
        cls.fact1Pool = diction['fact1']
        cls.fact2Pool = diction['fact2']
        cls.exprPool = diction['expr']
        cls.expr1Pool = diction['expr1']
        cls.y_aPool = diction['y_a']
        cls.implictionPool = diction['implication']

    @classmethod
    def geneNames(cls, n):
        """
        the strategy to select entities
        """
        random.seed(time())
        n = min(n, len(RandomGene.namePool))
        ret = list()
        tempPool = list(RandomGene.namePool)
        for i in range(n):
            s = RandomGene.geneFromInterval(0, len(tempPool) - 1)
            ret.append(tempPool[s])
            del tempPool[s]
        return ret

    @classmethod
    def geneAttrs(cls, n):
        """
        the strategy to select attributes
        """
        random.seed(time())
        n = min(n, len(RandomGene.attrPool))
        ret = list()
        tempPool = list(RandomGene.attrPool)
        for i in range(n):
            s = RandomGene.geneFromInterval(0, len(tempPool) - 1)
            ret.append(tempPool[s])
            del tempPool[s]
        return ret

    @classmethod
    def geneRelations(cls, n):
        """
        the strategy to select relations
        """
        random.seed(time())
        n = min(n, len(RandomGene.relationPool))
        ret = list()
        tempPool = list(RandomGene.relationPool)
        for i in range(n):
            s = RandomGene.geneFromInterval(0, len(tempPool) - 1)
            ret.append(tempPool[s])
            del tempPool[s]
        return ret

    @classmethod
    def geneFromInterval(cls, a, b):
        """
        a wrapped function to randomly select a num from [a,b]
        """
        # random.seed(time())
        # length = b - a + 1
        # rd = random.random()
        # kind = rd // (1 / length)
        # kind += a
        # assert kind <= b
        # return int(kind)
        return random.randint(a, b)

    @classmethod
    def geneFromInterval2(cls, a, b):
        """
        a wrapped function to randomly select two different num from [a,b]
        """
        length = b - a + 1
        rd = random.random()
        kind1 = rd // (1 / length)
        kind1 += a
        kind2 = kind1
        while int(kind2) == int(kind1):
            rd = random.random()
            kind2 = rd // (1 / length)
            kind2 += a
        return [int(kind1), int(kind2)]

    @classmethod
    def geneConstNum(cls):  # [1,4]
        """
        a wrapped function to randomly select num from [1,4]
        for D* datasets, we use this for linear operation
        and this is also used in some other settings
        """
        length = 4
        rd = random.random()
        kind = rd // (1 / length) + 1
        return int(kind)

    @classmethod
    def geneFact1Format(cls):
        rdsd = cls.geneFromInterval(0, len(cls.fact1Pool)-1)
        return cls.fact1Pool[rdsd]

    @classmethod
    def geneFact2Format(cls):
        rdsd = cls.geneFromInterval(0, len(cls.fact2Pool)-1)
        return cls.fact2Pool[rdsd]

    @classmethod
    def geneRule1Format(cls):
        rdsd = cls.geneFromInterval(0, len(cls.implictionPool)-1)
        implicationFormat = cls.implictionPool[rdsd]
        condition = cls.geneFact1Format().format(e='x', a='{a}', num='{numa}')
        conclusion = cls.geneFact1Format().format(e='x', a='{b}', num='{numb}')
        return implicationFormat.format(condition=condition, conclusion=conclusion)

    @classmethod
    def geneRule2Format(cls, k):
        rdsd = cls.geneFromInterval(0, len(cls.implictionPool)-1)
        implicationFormat = cls.implictionPool[rdsd]
        condition = cls.geneFact2Format().format(ei='x', ej='y', r='{r}')
        rdsd = cls.geneFromInterval(0, len(cls.y_aPool)-1)
        y_aFormat = cls.y_aPool[rdsd]
        if k == 1:
            rdsd = cls.geneFromInterval(0, len(cls.expr1Pool)-1)
            expr = cls.expr1Pool[rdsd]
        else:
            rdsd = cls.geneFromInterval(0, len(cls.exprPool)-1)
            expr = cls.exprPool[rdsd]
        expr = expr.format(y_a=y_aFormat, m='{m}', k='{k}')
        conclusion = cls.geneFact1Format().format(e='x', num=expr, a='{a}')
        return implicationFormat.format(condition=condition, conclusion=conclusion)


class Expr:
    int2str = ['zero', 'one', 'two', 'three', 'four',
               'five', 'six', 'seven', 'eight', 'nine', 'ten']

    def __init__(self):
        """
        mode = 0, ky+m
        """
        self.mode = 0
        self.argus = []
        if self.mode == 0:
            # k = [1,10], self.argus[0]
            self.argus.append(RandomGene.geneFromInterval(1, 10))
            # m = [1,10], self.argus[1]
            self.argus.append(RandomGene.geneFromInterval(1, 10))
            # if you want to generate D* datasets, you can use the codes below
            # self.argus.append(RandomGene.geneFromInterval(1, 2))
            # self.argus.append(RandomGene.geneFromInterval(1, 4))
        else:
            # add more expressions here
            raise 'Expr mode exceed'

    def compute(self, y):
        if self.mode == 0:
            return self.argus[0] * y + self.argus[1]
        else:
            # add more expressions here
            raise 'Expr mode exceed'

    def __str__(self):
        if self.mode == 0:
            return str(self.argus[0]) + "y+" + str(self.argus[1])
        else:
            # add more expressions here
            raise 'Expr mode exceed'

    # def nl(self, str):
    #     if self.mode == 0:
    #         if self.argus[0] == 1:
    #             return "y's " + str + " plus " + Expr.int2str[self.argus[1]]
    #         else:
    #             return "y's " + str + " multiplied by " + Expr.int2str[self.argus[0]] + " and then adding " + Expr.int2str[self.argus[1]]
    #     else:
    #         # add more expressions here
    #         raise 'Expr mode exceed'

    def expression_str(self, val):
        if self.mode == 0:
            if self.argus[0] == 1:
                return str(val) + '+' + str(self.argus[1]) + '=' + str(self.compute(val))
            else:
                return str(self.argus[0]) + '*' + str(val) + '+' + str(self.argus[1]) + '=' + str(self.compute(val))
        else:
            # add more expressions here
            raise 'Expr mode exceed'


class Fact:
    """
    kind = 0, is(e, a, num)
    kind = 1, r(e1, e2)
    """

    def __init__(self, entityNum, attrNum, relationNum, type=None):
        """
        the kind of this fact can be determined by type
        if type==None, then it is random
        we use the number to denote entity and attribute instead of string
        """
        if type is not None:
            self.kind = type
        else:
            self.kind = RandomGene.geneFromInterval(0, 1)
        self.content = dict()
        if self.kind == 0:  # is(e, a, num)
            self.content['entity'] = RandomGene.geneFromInterval(
                0, entityNum - 1)
            self.content['attr'] = (
                RandomGene.geneFromInterval(0, attrNum - 1))
            self.content['num'] = (RandomGene.geneConstNum())
        elif self.kind == 1:  # r(e1, e2)
            self.content['relation'] = (
                RandomGene.geneFromInterval(0, relationNum - 1))
            a, b = RandomGene.geneFromInterval2(0, entityNum - 1)
            self.content['entity1'] = a
            self.content['entity2'] = b
        else:  # more?
            raise 'factKindOutOfIndex'

    def __eq__(self, other):
        if self.kind != other.kind:
            return False
        elif self.kind == 0:
            return self.content['entity'] == other.content['entity'] and self.content['attr'] == other.content['attr']
        elif self.kind == 1:
            return self.content['relation'] == other.content['relation'] and ((self.content['entity1'] == other.content[
                'entity1'] and self.content['entity2'] == other.content['entity2']) or (self.content['entity1'] ==
                                                                                        other.content['entity2'] and
                                                                                        self.content['entity2'] ==
                                                                                        other.content['entity1']))
            # R(ei,ej)=R(ej,ei) It's not equal, but it's not allowed.
        else:
            raise 'factKindOutOfIndex'


class Rule:
    def __init__(self, attrNum, relationNum):
        """
        mode 0 is(x,a,numa) -> is(x,b,numb)
        mode 1 R(x,y) -> is(x,a,y,expr)
        contain = [dict, dict]
        """
        self.mode = RandomGene.geneFromInterval(0, 1)
        self.contain = list()
        if self.mode == 0:
            x = 0
            a, b = RandomGene.geneFromInterval2(0, attrNum - 1)
            numa = RandomGene.geneConstNum()
            self.contain.append({'x': x, 'attr': a, 'num': numa})
            numb = RandomGene.geneConstNum()
            self.contain.append({'x': x, 'attr': b, 'num': numb})
            assert a != b
        elif self.mode == 1:
            x, y = 0, 1
            r = RandomGene.geneFromInterval(0, relationNum - 1)
            a = RandomGene.geneFromInterval(0, attrNum - 1)
            expr = Expr()
            self.contain.extend([{'relation': r, 'x': x, 'y': y}, {
                'x': x, 'attr': a, 'y': y, 'expr': expr}])
        else:  # more?
            raise 'RuleModeIndexOutOfRange'

    def __eq__(self, other):  # we can't use it if true
        if self.mode != other.mode:
            return False
        elif self.mode == 0:
            if self.contain[0]['attr'] == other.contain[0]['attr'] and self.contain[0]['num'] == other.contain[0][
                    'num']:
                return True  # pure eq
            elif self.contain[0]['attr'] == other.contain[1]['attr'] and self.contain[1]['attr'] == other.contain[0][
                'attr'] and ((self.contain[0]['num'] == other.contain[1]['num']) or (
                    self.contain[1]['num'] == other.contain[0]['num'])):
                return True  # It's not equal, but it's not allowed.
        elif self.mode == 1:
            if self.contain[0]['relation'] == other.contain[0]['relation'] and self.contain[1]['attr'] == \
                    other.contain[1]['attr']:
                return True  # pure eq
        else:
            raise 'Rule__eq__ModeIndexOutOfRange'
        return False


class Assertion:
    def __init__(self, entity, attr):
        """
        mode 0 greater(e,a,num)
        mode 1 less(e,a,num)
        """
        self.mode = RandomGene.geneFromInterval(0, 1)
        self.content = {'entity': entity, 'attr': attr,
                        'num': RandomGene.geneFromInterval(1, 100)}


class Theory:
    def __init__(self, id, entityNum=4, attrNum=4, factNum=6, ruleNum=6, relationNum=3, depth=4):
        # debug
        self.debug_graph = ''
        self.debug_reason_iter = ''
        # pre prosses
        self.id = id
        self.names = RandomGene.geneNames(entityNum)
        self.attrs = RandomGene.geneAttrs(attrNum)
        self.relations = RandomGene.geneRelations(relationNum)
        self.entityNum = entityNum
        self.attrNum = attrNum
        self.factNum = factNum
        self.ruleNum = ruleNum
        self.relationNum = relationNum
        self.depth = depth + 1
        # prepare data structure
        self.facts = list()
        self.rules = list()
        self.inteq = list()
        self.intitution = list()
        self.assertion = None
        self.ans = None
        # helpers
        self.states = [[0 for _ in range(self.attrNum)]
                       for __ in range(self.entityNum)]
        # store all the attributes value of entities
        self.attr_num_graph = myGraph.Graph()
        self.entity_attr_graph = myGraph.Graph()
        # for detecting conflict
        self.reasoner = [[None for _ in range(self.attrNum)]
                         for __ in range(self.entityNum)]
        # store where does the value come from, for reasoning
        tryFactNum = 0  # record fact nums
        while tryFactNum < self.factNum:
            # [0,5] Control fact Species ratio
            rd = RandomGene.geneFromInterval(0, 5)
            if rd > 0:
                tmpFact = Fact(entityNum, attrNum, relationNum, 1)
            else:
                tmpFact = Fact(entityNum, attrNum, relationNum, 0)
            # Preliminary judgement of legality with the help of fact.eq
            if tmpFact not in self.facts:
                tryFactNum += 1
                self.facts.append(tmpFact)
                if tmpFact.kind == 0:  # e a num, write states
                    assert self.states[tmpFact.content['entity']
                                       ][tmpFact.content['attr']] == 0
                    self.states[tmpFact.content['entity']
                                ][tmpFact.content['attr']] = tmpFact.content['num']
                    assert self.reasoner[tmpFact.content['entity']
                                         ][tmpFact.content['attr']] is None
                    self.reasoner[tmpFact.content['entity']
                                  ][tmpFact.content['attr']] = [('F', len(self.facts) - 1)]
        tryRuleFact = 0

        while tryRuleFact < self.ruleNum:
            tmpRule = Rule(attrNum, relationNum)
            if tmpRule not in self.rules and self.noConflic(tmpRule):
                self.rules.append(tmpRule)
                tryRuleFact += 1

        self.default_reason()

        self.set_my_assertion()
        if self.assertion is not None:
            assert self.assertionSuitable()
            if self.serial_int(self.assertion.content['entity'], self.assertion.content['attr']) == False:
                self.assertion = None
                return
            self.derivation_int()

    def derivation_int(self):
        def tuple2str(tup):
            if tup[0] == 'F' and tup[1] == -1:
                return 'Default'
            elif tup[0] == 'F':
                return 'Fact' + str(tup[1])
            elif tup[0] == 'R':
                return 'Rule' + str(tup[1])
            elif tup[0] == 'S':
                intno = [no for no, iter in enumerate(
                    self.inteq) if iter == tup[1:]]
                assert len(intno) == 1, "there should be exact one intitution"
                return 'int' + str(intno[0])
            else:
                raise "tuple convert fault on {0}".format(tup)

        def reason2str(reason):
            lis_reason = [tuple2str(tup) for tup in reason]
            return ' & '.join(lis_reason)

        for intno, (entity, attr) in enumerate(self.inteq):
            intitution_content = self.names[entity] + ' is '
            if len(self.reasoner[entity][attr]) <= 2:
                intitution_content += str(self.states[entity][attr])
            else:
                depend_e, depend_a = self.reasoner[entity][attr][2][1:]
                intitution_content += self.rules[self.reasoner[entity][attr][1][1]].contain[1]['expr'].expression_str(
                    self.states[depend_e][depend_a])
            intitution_content += ' ' + self.attrs[attr]
            self.intitution.append(
                reason2str(self.reasoner[entity][attr]) + ' -> int' + str(intno) + ': ' + intitution_content)

    def serial_int(self, e, a):
        q = Queue()
        intdict = dict()
        iter = (e, a)
        cnt = 0
        iter_set = set()
        while True:
            self.debug_reason_iter += str(iter) + '\n'
            if iter in iter_set:
                return False
            iter_set.add(iter)
            q.put(iter)
            intdict[iter] = cnt
            cnt += 1
            reason = self.reasoner[iter[0]][iter[1]]
            if len(reason) == 1:
                break
            iter = reason[-1][1:]
        self.inteq = [-1] * cnt
        cnt -= 1
        while not q.empty():
            iter = q.get()
            intdict[iter] = cnt - intdict[iter]
            self.inteq[intdict[iter]] = iter
        return True

    def default_reason(self):
        for entity, entity_lis in enumerate(self.reasoner):
            for attr, attr_reason in enumerate(entity_lis):
                if attr_reason is None:
                    assert self.states[entity][attr] == 0
                    self.reasoner[entity][attr] = [
                        ('F', -1)]  # which is default

    def set_my_assertion(self):
        def nested_list_depth(lst):
            if not isinstance(lst, list):
                return 0
            if not lst:
                return 1
            return 1 + max(nested_list_depth(item) for item in lst)

        tmpfact = Fact(self.entityNum, self.attrNum, self.relationNum, 0)
        tmpfact.kind = 0
        # ensure assertion is not easy
        preliminarySolutions = []
        for e, attrList in enumerate(self.states):
            for a, num in enumerate(attrList):
                if num != 0 and self.serial_int(e, a) and self.depth == nested_list_depth(self.do_lis_reasoning(e, a)):
                    tmpfact.content = {'entity': e, 'attr': a}
                    if tmpfact not in self.facts:
                        preliminarySolutions.append((e, a))
        if len(preliminarySolutions) == 0:
            return
        soluNo = RandomGene.geneFromInterval(0, len(preliminarySolutions) - 1)
        e, a = preliminarySolutions[soluNo]
        self.assertion = Assertion(e, a)

    def assertionSuitable(self):
        """
        non-zero and non-fact declared
        """
        e = self.assertion.content['entity']
        a = self.assertion.content['attr']
        if self.states[e][a] == 0:
            return False

        tmpfact = Fact(self.entityNum, self.attrNum, self.relationNum, 0)
        tmpfact.kind = 0
        tmpfact.content = {'entity': e, 'attr': a}
        if tmpfact in self.facts:
            return False

        if self.assertion.mode == 0:
            self.ans = self.states[e][a] > self.assertion.content['num']
        else:
            self.ans = self.states[e][a] < self.assertion.content['num']
        return True

    def dfs_modify(self, ent, atr, val, state, reasoner, reason):
        if state[ent][atr] != 0:
            return False
        assert reasoner[ent][atr] is None
        reasoner[ent][atr] = reason
        state[ent][atr] = val
        # dfs
        for ruleno, tmpRule in enumerate(self.rules):
            if tmpRule.mode == 0 and tmpRule.contain[0]['attr'] == atr and tmpRule.contain[0]['num'] == val:
                new_reason = [('R', ruleno), ('S', ent, atr)]
                if not self.dfs_modify(ent, tmpRule.contain[1]['attr'], tmpRule.contain[1]['num'], state, reasoner,
                                       new_reason):
                    return False
            if tmpRule.mode == 1 and tmpRule.contain[1]['attr'] == atr:
                for factno, tmpFact in enumerate(self.facts):
                    if tmpFact.kind == 1 and tmpFact.content['relation'] == tmpRule.contain[0]['relation'] and \
                            tmpFact.content['entity2'] == ent:
                        x = tmpFact.content['entity1']
                        state[x][atr] = 0
                        new_reason = [
                            ('F', factno), ('R', ruleno), ('S', ent, atr)]
                        assert reasoner[x][atr] == new_reason, "reasoner:{0}\n new_reason:{1}".format(reasoner[x][atr],
                                                                                                      new_reason)
                        reasoner[x][atr] = None
                        if not self.dfs_modify(x, atr, tmpRule.contain[1]['expr'].compute(val), state, reasoner,
                                               new_reason):
                            return False
        return True

    def noConflic(self, tmpRule):
        tmpStates = copy.deepcopy(self.states)
        tmpReasoner = copy.deepcopy(self.reasoner)
        if tmpRule.mode == 0:
            attra = tmpRule.contain[0]['attr']
            numa = tmpRule.contain[0]['num']
            attrb = tmpRule.contain[1]['attr']
            numb = tmpRule.contain[1]['num']
            for number, entity in enumerate(tmpStates):
                if entity[attra] == numa and entity[attrb] != 0:
                    return False
                elif entity[attra] == numa and entity[attrb] == 0:
                    reason = [('R', len(self.rules)), ('S', number, attra)]
                    if not self.dfs_modify(number, attrb, numb, tmpStates, tmpReasoner, reason):
                        return False
            self.attr_num_graph.addEdge(attra, numa, attrb, numb)
            if self.attr_num_graph.cycleDetect():
                self.attr_num_graph.deleteEdge(attra, numa, attrb, numb)
                return False
        elif tmpRule.mode == 1:
            relation = tmpRule.contain[0]['relation']
            tmpgraph = copy.deepcopy(self.entity_attr_graph)
            debug_graph_tmp = ''
            for facno, fac in enumerate(self.facts):
                if fac.kind == 1 and fac.content['relation'] == relation:
                    e1 = fac.content['entity1']
                    e2 = fac.content['entity2']
                    attr = tmpRule.contain[1]['attr']
                    if tmpStates[e1][attr] != 0:
                        return False
                    else:
                        reason = [('F', facno), ('R', len(
                            self.rules)), ('S', e2, attr)]
                        if not self.dfs_modify(e1, attr, tmpRule.contain[1]['expr'].compute(tmpStates[e2][attr]),
                                               tmpStates, tmpReasoner, reason):
                            return False
                    tmpgraph.addEdge(e2, attr, e1, attr)
                    debug_graph_tmp += '\nadd edge - attr: {0}, e1: {1}, e2: {2}, r: {3}'.format(
                        attr, e1, e2, relation)
                    if tmpgraph.cycleDetect():
                        tmpgraph.deleteEdge(e2, attr, e1, attr)
                        debug_graph_tmp += '\ndel edge - attr: {0}, e1: {1}, e2: {2}'.format(
                            attr, e1, e2)
                        return False
            self.entity_attr_graph = tmpgraph
            self.debug_graph += debug_graph_tmp
        self.states = copy.deepcopy(tmpStates)
        self.reasoner = copy.deepcopy(tmpReasoner)
        return True

    def debug_info(self):
        info = ''
        for no, name in enumerate(self.names):
            info += 'name' + str(no) + ': ' + name + '\n'
        for no, attr in enumerate(self.attrs):
            info += 'attr' + str(no) + ': ' + attr + '\n'
        for no, rela in enumerate(self.relations):
            info += 'relation' + str(no) + ': ' + rela + '\n'
        info += str(self.reasoner) + '\n'
        info += self.debug_reason_iter + '\n'
        info += self.debug_graph + '\n'
        return info

    def __str__(self):
        return "ID: {0}\n".format(self.id) + self.str_facts() + self.str_rules() + self.str_assertion() + \
            self.answer() + self.list_reason() + self.str_reason()

    def nl(self):
        return self.facts_nl() + self.rules_nl() + self.assertion_nl() + self.str_reason() + self.answer()

    def str_facts(self):
        ret = "Fact:\n"
        for i, fact in enumerate(self.facts):
            ret += str(i) + ". "
            if fact.kind == 0:
                ret += "is(" + self.names[fact.content['entity']] + ", " + \
                       self.attrs[fact.content['attr']] + \
                       ", " + str(fact.content['num']) + ")\n"
            elif fact.kind == 1:
                ret += self.relations[fact.content['relation']] + \
                    "(" + self.names[fact.content['entity1']] + ", " + \
                    self.names[fact.content['entity2']] + ")\n"
        return ret

    def facts_nl(self):
        ret = "Fact:\n"
        for i, fact in enumerate(self.facts):
            ret += str(i) + ". "
            if fact.kind == 0:
                template = RandomGene.geneFact1Format().format(e=self.names[fact.content['entity']], num=str(
                    fact.content['num']), a=self.attrs[fact.content['attr']])
            elif fact.kind == 1:
                template = RandomGene.geneFact2Format().format(
                    r=self.relations[fact.content['relation']], ei=self.names[fact.content['entity1']], ej=self.names[fact.content['entity2']])
            if template and template[0].islower():
                template = template[0].upper() + template[1:]
            ret += template + ".\n"
        return ret

    def rules_nl(self):
        ret = "Rule:\n"
        for i, rule in enumerate(self.rules):
            ret += str(i) + ". "
            if rule.mode == 0:
                template = RandomGene.geneRule1Format().format(a=self.attrs[rule.contain[0]['attr']], numa=str(
                    rule.contain[0]['num']), b=self.attrs[rule.contain[1]['attr']], numb=str(rule.contain[1]['num']))
            elif rule.mode == 1:
                template = RandomGene.geneRule2Format(rule.contain[1]['expr'].argus[0]).format(
                    r=self.relations[rule.contain[0]['relation']], a=self.attrs[rule.contain[1]['attr']], k=str(rule.contain[1]['expr'].argus[0]), m=str(rule.contain[1]['expr'].argus[1]))
            if template and template[0].islower():
                template = template[0].upper() + template[1:]
            ret += template + ".\n"
        return ret

    def str_rules(self):
        ret = "Rule:\n"
        for i, rule in enumerate(self.rules):
            ret += str(i) + ". "
            if rule.mode == 0:
                ret += "is(?x, " + self.attrs[rule.contain[0]['attr']] + ", " + str(rule.contain[0]['num']) + ") -> " + \
                       "is(?x, " + self.attrs[rule.contain[1]['attr']
                                              ] + ", " + str(rule.contain[1]['num']) + ")\n"
            elif rule.mode == 1:
                ret += self.relations[rule.contain[0]['relation']] + \
                    "(?x, ?y) -> is(?x, " + \
                    self.attrs[rule.contain[1]['attr']] + \
                    ", " + str(rule.contain[1]['expr']) + ")\n"
        return ret

    def str_assertion(self):
        ret = "Assertion:\n"
        if self.assertion.mode == 0:  # greater(e,a,num)
            ret += "greater("
        else:
            ret += "less("
        ret += self.names[self.assertion.content['entity']] + ", " + \
            self.attrs[self.assertion.content['attr']] + \
            ", " + str(self.assertion.content['num']) + ")\n"
        return ret

    def assertion_nl(self):
        ret = "Assertion:\n"
        ret += self.names[self.assertion.content['entity']] + \
            "'s " + self.attrs[self.assertion.content['attr']]
        if self.assertion.mode == 0:  # greater(e,a,num)
            ret += " is greater than "
        else:
            ret += " is less than "
        ret += str(self.assertion.content['num']) + ".\n"
        return ret

    def answer(self):
        ret = "Answer:\n"
        if self.ans:
            ret += "True\n"
        else:
            ret += "False\n"
        return ret

    def list_reason(self):
        ret = "One reasoning:\n"
        reason_lis = self.do_lis_reasoning(
            self.assertion.content['entity'], self.assertion.content['attr'])
        ret += str(reason_lis) + '\n'
        return ret

    def str_reason(self):
        ret = "Step reasoning:\n"
        reason_steps = self.step_reasoning()
        ret += reason_steps + '\n'
        return ret

    def to_json(self) -> dict:
        ret = {}
        ret['id'] = self.id
        ret['depth'] = self.depth - 1
        ret['fact_num'] = self.factNum
        ret['rule_num'] = self.ruleNum
        ret['label'] = 1 if self.ans else 0

        ret['facts'] = self.str_facts()
        ret['rules'] = self.str_rules()
        ret['assertion'] = self.str_assertion()
        ret['answer'] = self.answer()

        ret['list_reason'] = self.list_reason()
        ret['str_reason'] = self.str_reason()

        ret['answer_nl'] = self.answer()
        ret['facts_nl'] = self.facts_nl()
        ret['rules_nl'] = self.rules_nl()
        ret['assertion_nl'] = self.assertion_nl()
        return ret

    def do_lis_reasoning(self, e, a):
        def recursion(reason):
            ret = []
            for tup in reason:
                ret.append(tuple2str(tup))
            return ret

        def tuple2str(tup):
            if tup[0] == 'F' and tup[1] == -1:
                return 'Default'
            elif tup[0] == 'F':
                return 'Fact' + str(tup[1])
            elif tup[0] == 'R':
                return 'Rule' + str(tup[1])
            elif tup[0] == 'S':
                return recursion(self.reasoner[tup[1]][tup[2]])
            else:
                raise "tuple convert fault on {0}".format(tup)

        return recursion(self.reasoner[e][a])

    def step_reasoning(self):
        return '; '.join(self.intitution)
