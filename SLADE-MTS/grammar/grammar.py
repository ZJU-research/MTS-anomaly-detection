from .symbol import Symbol
from .rule import Rule
from .segment import Segment

class Grammar(object):
    """docstring for Grammar"""
    def __init__(self):
        super(Grammar, self).__init__()
        self.digram_index = {}
        self.root_production = Rule(self)
        self.segments = []

        self.count = 0

    def train_string(self, string):
        """docstring for train_string"""
        self.segments.extend(string)
        input_sequence = [c for c in string]
        if (0 < len(input_sequence)):
            self.root_production.last().insert_after(Symbol.factory(self, input_sequence.pop(0)))
        while (0 < len(input_sequence)):
            self.root_production.last().insert_after(Symbol.factory(self, input_sequence.pop(0)))
            match = self.get_index(self.root_production.last().prev)
            if not match:
                self.add_index(self.root_production.last().prev)
            elif match.next != self.root_production.last().prev:
                self.root_production.last().prev.process_match(match)


    def add_index(self, digram):
        """docstring for index"""
        self.digram_index[digram.hash_value()] = digram

    def get_index(self, digram):
        """docstring for get"""
        return self.digram_index.get(digram.hash_value())

    def clear_index(self, digram):
        """docstring for clear_index"""
        if self.digram_index.get(digram.hash_value()) == digram:
            self.digram_index[digram.hash_value()] = None

    def print_grammar(self):
        """docstring for print_grammar"""
        output_array = []
        rule_set = [self.root_production]
    
        i = 0
        for rule in rule_set:
            output_array.append("%s --(%d)--> " % (i, rule.reference_count))
            line_length = rule.print_rule(rule_set, output_array, len("%s --(%d)--> " % (i, rule.reference_count)))
            output_array.append('\n')
            i += 1
        return "".join(output_array)

    def get_rule_set(self):
        """get_rule_set"""
        output_array = []
        self.count = 0
        rule_set = [self.root_production]
        line_length = 0
        self.root_production.print_rule_expansion(rule_set, output_array, line_length, [self.root_production]) #set segment's rule
    
        i = 0
        for rule in rule_set:
            output_array.append("%s --(%d)--> " % (i, rule.reference_count))
            line_length = rule.print_rule(rule_set, output_array, len("%s --(%d)--> " % (i, rule.reference_count)))
            output_array.append('\n')
            i += 1

        return rule_set
