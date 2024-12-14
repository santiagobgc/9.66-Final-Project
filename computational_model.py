
import random
import treys
import pymc as pm
from treys import Deck
from treys import Card
import numpy as np
from itertools import combinations 

class Player:
    def __init__(self, name, money):
        self.name = name
        self.money = money
        self.cards = []
        self.in_hand = True  # Whether the player is still in the hand
        self.total_bet = 0
        self.history = []

    def reset(self):
        self.cards = []
        self.in_hand = True
        self.total_bet = 0
        self.history = []

    def make_bet(self, amount):
        self.total_bet += amount
        self.money -= amount


    def evaluate_hand(self, board):
        evaluator = treys.Evaluator()
        return evaluator.evaluate(board, self.cards)

class GutsGame:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.players = [player1, player2]
        self.pot = 0
        self.current_bet = 0
        self.current_player_index = 0
        self.deck = treys.Deck()
        self.board = self.deck.draw(5)
        self.round_id = 0


    def shuffle_deck(self):
        self.deck = treys.Deck()

    def deal_cards(self):
        self.board = self.deck.draw(5)
        for player in self.players:
            player.reset()
            player.cards = self.deck.draw(2)
            Card.print_pretty_cards(player.cards)
            print(player.evaluate_hand(self.board))
        Card.print_pretty_cards(self.board)

    def raise_bet(self, player, amount):
        amount_added = self.current_bet + amount - player.total_bet
        self.pot += amount_added
        self.current_bet += amount
        player.make_bet(amount_added)

    def call(self, player):
        amount_added = self.current_bet - player.total_bet
        self.pot += amount_added
        player.make_bet(amount_added)

    def fold(self, player):
        player.in_hand = False

    def rotate_starting_player(self):
        self.current_player_index = (self.current_player_index + 1) % len(self.players)

    
    # def real_game(self, board, player1hand, player2hand):
    #     # self.shuffle_deck()
    #     self.board = board
    #     for player in self.players:
    #         player.reset()
    #     self.player1.cards = player1hand
    #     self.player2.cards = player2hand
    #     #all_used = board + player1hand + player2hand
    #     for card in board:
    #         if card in self.deck.cards:
    #             self.deck.cards.remove(card)
    #         else:
    #             print(card)
    #             print(self.deck.cards)
    #     Card.print_pretty_cards(self.board)


    def win_prob(self, player):
        evaluator = treys.Evaluator()
        test_deck = treys.Deck()
        for card in player.cards + self.board:
            test_deck.cards.remove(card)
        pos_opponent_hands = list(combinations(test_deck.cards, 2))

        hand_strength = player.evaluate_hand(self.board)
        wins = 0
        ties = 0
        losses = 0
        all_strengths = []
        for opponent_hand in  pos_opponent_hands:
            opponent_strength = evaluator.evaluate(self.board, list(opponent_hand))
            all_strengths.append(opponent_strength)
            if hand_strength < opponent_strength:
                wins += 1
            elif hand_strength == opponent_strength:
                ties += 1
            else:
                losses += 1
        all_strengths = np.array(all_strengths)
        all_probabilities = [1/len(all_strengths)]*len(all_strengths)
        all_probabilities = np.array(all_probabilities)

        total_scenarios = len(pos_opponent_hands)
        win_prob = wins / total_scenarios
        tie_prob = ties / total_scenarios
        loss_prob = losses / total_scenarios
        return (win_prob, tie_prob, loss_prob,all_strengths,all_probabilities)

    def call_update(self, player, hands, probabilities):
        q1, q3 = np.percentile(hands, [25, 75])

        q1_indices = hands > q3
        q4_indices = hands < q1
        q2_q3_indices = ~q1_indices & ~q4_indices

        lost_prob = np.sum(probabilities[q1_indices]) / 2 + np.sum(probabilities[q4_indices]) / 2

        probabilities[q1_indices] /= 2
        probabilities[q4_indices] /= 2

        if np.sum(q2_q3_indices) > 0:
            probabilities[q2_q3_indices] += lost_prob / np.sum(q2_q3_indices)

        probabilities /= np.sum(probabilities)

        assert np.isclose(np.sum(probabilities), 1.0, atol=1e-10), f"Probabilities sum to {np.sum(probabilities)}"
        return probabilities
        
    def raise_update(self, player, hands, probabilities):
        q1, q3 = np.percentile(hands, [25, 75])
        median = np.median(hands)

        q1_q2_indices = hands > median
        q3_indices = (hands >= q1) & (hands <= median)
        q4_indices = hands < q1

        lost_prob = (
            np.sum(probabilities[q1_q2_indices]) * (3 / 4)
            + np.sum(probabilities[q3_indices]) * (1 / 2)
        )

        probabilities[q1_q2_indices] *= (1 / 4)
        probabilities[q3_indices] *= (1 / 2)

        if np.sum(q4_indices)>0:
            probabilities[q4_indices] += lost_prob / np.sum(q4_indices)
        
        probabilities /= np.sum(probabilities)

        assert np.isclose(np.sum(probabilities), 1.0, atol=1e-10), f"Probabilities sum to {np.sum(probabilities)}"


        return probabilities


    def player_action(self, player):
        if not player.in_hand or player.money <= 0:
            self.fold(player)
            return "fold"

        start_win_prob, start_tie_prob, start_lose_prob, all_hands, all_probabilities = self.win_prob(player)

        for opponent in self.players:
            if opponent != player:
                opp = opponent
                for action in opponent.history:
                    if action == "call":
                        all_probabilities = self.call_update(opponent, all_hands, all_probabilities)
                    elif action == "raise":
                            all_probabilities = self.raise_update(opponent, all_hands, all_probabilities)
                        
        player_hand_strength = player.evaluate_hand(self.board)
        win_prob = np.sum(all_probabilities[all_hands > player_hand_strength])
        tie_prob = np.sum(all_probabilities[all_hands == player_hand_strength])
        lose_prob = 1 - (win_prob + tie_prob)  
        bluff_prob = 0.2 / opp.history.count("raise") if opp.history.count("raise") > 0 else 0
        fold_ev = 0 - bluff_prob * self.pot
        call_ev = (self.pot) * win_prob - (self.current_bet - player.total_bet) * lose_prob
        raise_ev = (self.pot + self.current_bet) * win_prob - (2 * self.current_bet - player.total_bet) * (lose_prob +bluff_prob)
        print(f"Start win Probability: {start_win_prob}, start Tie Probability: {start_tie_prob}, start Losing Probability: {start_lose_prob}")
        print(f"Winning Probability: {win_prob}, Tie Probability: {tie_prob}, Losing Probability: {lose_prob}")
        print(f"Raise: {raise_ev}, Call: {call_ev}, Fold: {fold_ev}")
        print(opp.history)
        if raise_ev > call_ev and raise_ev > fold_ev:
            if player.money >= 2*self.current_bet:
                self.raise_bet(player, self.current_bet)
                player.history.append("raise")
                print(player.name, "raise")
                return "raise"
            elif player.money >= self.current_bet:
                self.call(player)
                player.history.append("call")
                print(player.name, "call")
                return "call"
        elif call_ev > raise_ev and call_ev > fold_ev:
            if player.money >= self.current_bet:
                self.call(player)
                player.history.append("call")
                print(player.name, "call")
                return "call"
            else:
                self.fold(player)
                player.history.append('fold')
                print(player.name, "fold")
                return "fold"
        else:  
            self.fold(player)
            player.history.append("fold")
            print(player.name, "fold")
            return "fold"


    def determine_winner(self):
        active_players = [p for p in self.players if p.in_hand]
        if not active_players:
            return None

        winner = min(active_players, key=lambda p: p.evaluate_hand(self.board))
        winner.money += self.pot
        self.pot = 0
        return winner

    def play_round(self):
        self.round_id += 1
        self.shuffle_deck()
        self.deal_cards()
        #self.real_game(boards[number], santi_hands[number], jack_hands[number])
        self.pot = 0
        self.current_bet = 0

        starting_player = self.players[self.current_player_index]
        if starting_player.money >= 1:
            self.raise_bet(starting_player, 1)
        first_turn = True
        curr_player = self.players[1-self.current_player_index]
        curr_idx = 1 - self.current_player_index
        new_bet = False
        while True:
            new_bet = False
            action = self.player_action(curr_player)
            if action == "raise":
                new_bet = True
                curr_idx = 1 - curr_idx
                curr_player = self.players[curr_idx]
            if not new_bet:
                if not first_turn or action =='fold':
                    break
                else:
                    first_turn = False
                    curr_idx = 1 - curr_idx
                    curr_player = self.players[curr_idx]
            first_turn = False

        winner = self.determine_winner()
        print(f"Winner of this round: {winner.name} with ${winner.money}")
        self.rotate_starting_player()
        return winner

    def play_game(self):
        while all(p.money > 0 for p in self.players):
            print(f"Starting a new round. Pot: ${self.pot}")
            winner = self.play_round()
            
            if winner:
                print(f"{winner.name} wins the pot! Current money: {winner.money}")

        winner = max(self.players, key=lambda p: p.money)
        print(f"Game over! The winner is {winner.name} with ${winner.money}")

# player1 = Player("Alice", 10)
# player2 = Player("Bob", 10)
# game = GutsGame(player1,player2)
# game.play_game()

#game data
import treys
import pymc as pm
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from treys import Deck, Evaluator, Card
from itertools import combinations

santi_winprobs = [0.2, 0.1, 0.7, 0.15, 0.25, 0.4, 0.9, 0.3, 0.1, 0.2, 0.1, 0.85, 0.6, 0.15, 0.25,
                  0.55, 0.9, 0.55, 0.8, 0.99, 0.75, 0.85, 0.3, 0.4, 0.25, 0.25, 0.55, 0.6, 0.15, 0.95, 0.1, 0.95]
santi_money = [10,11,9,9,9,5,7,1,10,10,9,9,9,10,8,9,9,13,14,16,18,19,19,19,18,16,16,17,17,15,19,19]
santi_hands = []
santi_hands.append([Card.new('Qc'), Card.new('9s')])
santi_hands.append([Card.new('4c'), Card.new('7c')])
santi_hands.append([Card.new('Qs'), Card.new('2s')])
santi_hands.append([Card.new('6h'), Card.new('7c')])
santi_hands.append([Card.new('2c'), Card.new('7s')])
santi_hands.append([Card.new('Ad'), Card.new('Tc')])
santi_hands.append([Card.new('Jd'), Card.new('3c')])
santi_hands.append([Card.new('9h'), Card.new('Jh')])

jack_winprobs = [0.1, 0.80, 0.03, 0.9, 0.6, 0.02, 0.5, 0.5, 0.25, 0.75, 0.8, 0.2, 0.35, 0.5,0.02, 0.35, 0.7,
                 0.03, 0.4, 0.05, 0.30, 0.50, 0.97, 0.97, 0.3, 0.05, 0.02, 0.05, 0.99, 0.65, 0.5, 0.6]
jack_money = [10,9,11,11,11,15,13,19,10,10,11,11,11,10,12,11,11,7,6,4,2,1,1,1,2,4,4,3,3,5,1,1]
jack_hands = []
jack_hands.append([Card.new('Th'), Card.new('7s')])
jack_hands.append([Card.new('6c'), Card.new('6s')])
jack_hands.append([Card.new('8h'), Card.new('2d')])
jack_hands.append([Card.new('8c'), Card.new('2d')])
jack_hands.append([Card.new('Ks'), Card.new('3d')])
jack_hands.append([Card.new('7c'), Card.new('6c')])
jack_hands.append([Card.new('2c'), Card.new('6s')])
jack_hands.append([Card.new('Kh'), Card.new('4d')])


boards = []
boards.append([Card.new('6d'), Card.new('Jd'), Card.new('2h'), Card.new('Jh'), Card.new('Ks')])
boards.append([Card.new('9d'), Card.new('Js'), Card.new('Kd'), Card.new('6h'), Card.new('Ts')])
boards.append([Card.new('Qd'), Card.new('7d'), Card.new('6c'), Card.new('6d'), Card.new('Js')])
boards.append([Card.new('5c'), Card.new('Js'), Card.new('Tc'), Card.new('2c'), Card.new('2h')])
boards.append([Card.new('3s'), Card.new('5c'), Card.new('4s'), Card.new('Jh'), Card.new('4h')])
boards.append([Card.new('9s'), Card.new('Qc'), Card.new('Kc'), Card.new('3s'), Card.new('8h')])
boards.append([Card.new('Jc'), Card.new('4c'), Card.new('8c'), Card.new('6h'), Card.new('3s')])
boards.append([Card.new('4s'), Card.new('3h'), Card.new('8d'), Card.new('7d'), Card.new('Ac')])

game2 = [
    ("QD,6H,6C,KS,TD", "8S,2S", "JC, 5C"),
    ("8C,TD,AD,4C,7D", "QS,5D", "AC, 6D"),
    ("QD,9H,AH,KH,JD", "8H,3S", "AC, QH"),
    ("JD,8D,6C,QD,7H", "JC,QC", "KH, 9C"),
    ("6C,TH,JH,2D,TS", "8S,8D", "AH, 5D"),
    ("9H,7H,QH,7D,KH", "6C,TD", "9C, 3D"),
    ("2H,QD,TD,QH,QC", "AD,7C", "3D, 5D"),
    ("JS,AC,6D,8S,8C", "6H,3S", "KC, 3H"),
    ("3H,4S,QC,8D,QH", "QD,7H", "8C, 5H"),
    ("KD,4H,TC,QC,9H", "8H,9C", "8S, 5C"),
    ("JH,8D,9H,8S,5C", "KS,JC", "AC, KC"),
    ("TS,TC,7C,QC,6H", "7H,TD", "9D, 3S"),
    ("5C,9H,AC,6H,8S", "AH,4H", "KC, JC"),
    ("2H,QD,7C,TC,4C", "7D,TD", "8D, 2S"),
    ("2D,2C,8D,3H,9D", "AH,4D", "JD, 7D"),
    ("AC,2S,JD,TH,JH", "4H,2H", "JC, 8S"),
    ("6D,KS,QS,3S,JS", "3H,2D", "7D, 6C"),
    ("7D,JS,8S,5H,5C", "AS,3C", "TH, 2H"),
    ("9S,5S,KD,TS,AH", "9D,QS", "6H, 4C"),
    ("TC,5D,7H,8H,2D", "QH,5H", "9C, 3S"),
    ("AC,3D,6S,9C,5D", "QC,2D", "4H, 2H"),
    ("4D,4C,7C,2C,JH", "9C,3C", "7S, 9H"),
    ("9D,JC,2H,6H,4D", "8C,5D", "2C, 7H"),
    ("KC,2H,7C,7D,4D", "7H,5H", "3H, 4C"),
]

player1 = Player("Santi", 10)
player2 = Player("Jack", 10)
game = GutsGame(player1,player2)
game.play_game()
