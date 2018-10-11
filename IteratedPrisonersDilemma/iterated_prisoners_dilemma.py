"""
Code for the Iterated Prisoner's Dilemma
from exercise 2.4 of Homework 2
from the course Multi-Agent Systems
"""
import random
from enum import Enum
from typing import List


class Choice(Enum):
    CONFESS = 0
    DENY = 1

class Player:
    def __init__(self, choice=Choice.CONFESS):
        self.choice: Choice = choice
        self.past_choices = [choice]

    def set_choice(self, choice: Choice):
        self.choice = choice
    
    def add_to_past(self):
        self.past_choices.append(self.choice)

    def next_choice(self, opponent_choice_list: List[Choice]):
        raise NotImplementedError("Please implement this method!")


class AlwaysConfessPlayer(Player):
    """
    Player that always confesses
    """
    def __init__(self):
        Player.__init__(self)

    def next_choice(self, opponent_choice_list: List[Choice]):
        self.set_choice(Choice.CONFESS)


class AlwaysDenyPlayer(Player):
    """
    Player that always denies
    """
    def __init__(self):
        Player.__init__(self, Choice.DENY)

    def next_choice(self, opponent_choice_list: List[Choice]):
        self.set_choice(Choice.DENY)


class RandomPlayer(Player):
    """
    Player that plays a random move each game
    """
    def __init__(self):
        Player.__init__(self)
        self.choice = random.choice(list(Choice))

    def next_choice(self, opponent_choice_list: List[Choice]):
        new_choice = random.choice(list(Choice))
        self.set_choice(new_choice)


class TitForTatPlayer(Player):
    """
    Player that confesses on the first move,
    then copies the opponent's move.
    """
    def __init__(self):
        Player.__init__(self)

    def next_choice(self, opponent_choice_list: List[Choice]):
        last_choice = opponent_choice_list[-1]
        self.set_choice(last_choice)


class ReverseTitForTatPlayer(Player):
    """
    Player that denies on the first move,
    the plays the opposite of the opponent's move.
    """
    def __init__(self):
        Player.__init__(self, Choice.DENY)

    def next_choice(self, opponent_choice_list: List[Choice]):
        last_choice = opponent_choice_list[-1]
        if last_choice == Choice.CONFESS:
            self.set_choice(Choice.DENY)
        else:
            self.set_choice(Choice.CONFESS)


class GrimTrigger(Player):
    """
    Player that confesses, until the opponent denies, and thereafter always denies.
    """
    def __init__(self):
        Player.__init__(self)

    def next_choice(self, opponent_choice_list: List[Choice]):
        if Choice.DENY in opponent_choice_list:
            self.set_choice(Choice.DENY)
        else:
            self.set_choice(Choice.CONFESS)


class Game:
    def __init__(self):
        # initialize pay-off matrix
        self.pay_off_matrix = [[0 for x in range(2)] for y in range(2)]
        self.pay_off_matrix[0][0] = (-1, -1)
        self.pay_off_matrix[0][1] = (-12, 0)
        self.pay_off_matrix[1][0] = (0, -12)
        self.pay_off_matrix[1][1] = (-8, -8)
        # initialize players
        self.player1 = ReverseTitForTatPlayer()
        self.player2 = GrimTrigger()

    def iterated_game(self, N=100):
        player1_score = 0
        player2_score = 0
        for _ in range(N):
            x, y = self.result()
            player1_score += x
            player2_score += y
            self.player1.add_to_past()
            self.player2.add_to_past()
            self.player1.next_choice(self.player2.past_choices)
            self.player2.next_choice(self.player1.past_choices)
        
        return player1_score, player2_score

    def result(self) -> tuple:
        return self.pay_off_matrix[self.player1.choice.value][self.player2.choice.value]

    def print_matrix(self):
        for row in self.pay_off_matrix:
            for val in row:
                print(val, end=" ")
            print()

game = Game()
win_1 = 0
draw = 0
win_2 = 0
for _ in range(100):
    p1, p2 = game.iterated_game()
    if p1 > p2:
        win_1 += 1
    elif p1 == p2:
        draw += 1
    else:
        win_2 += 1
print("Player 1 score:", win_1 / 100)
print("Draw percentage:", draw / 100)
print("Player 2 score:", win_2 / 100)
