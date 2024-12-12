
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
        """Evaluate the strength of the hand. Higher value means stronger hand."""
        # Simplified: sum of card values (can be replaced with more advanced logic)
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

        q1_indices = hands < q1
        q4_indices = hands > q3
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

        q1_q2_indices = hands < median
        q3_indices = (hands >= median) & (hands <= q3)
        q4_indices = hands > q3

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
        win_prob = np.sum(all_probabilities[all_hands < player_hand_strength])
        tie_prob = np.sum(all_probabilities[all_hands == player_hand_strength])
        lose_prob = 1 - (win_prob + tie_prob)  
        bluff_prob = 0.2 / opp.history.count("raise") if opp.history.count("raise") > 0 else 0
        fold_ev = 0 - bluff_prob * self.pot
        call_ev = (self.pot) * win_prob - (self.current_bet - player.total_bet) * lose_prob
        raise_ev = (self.pot + self.current_bet) * win_prob - (2 * self.current_bet - player.total_bet) * (lose_prob +bluff_prob)
        print(f"Winning Probability: {win_prob}, Tie Probability: {tie_prob}, Losing Probability: {lose_prob}")
        print(f"Raise: {raise_ev}, Call: {call_ev}, Fold: {fold_ev}")
        if win_prob > 0.6: 
            if player.money >= self.current_bet + 1:
                self.raise_bet(player, 1)
                player.history.append("raise")
                print("R")
                return "raise"
        elif win_prob > 0.3:  # Example threshold for calling
            if player.money >= self.current_bet:
                self.call(player)
                player.history.append("call")
                print('C')
                return "call"
        else:  # Fold if probabilities are low
            self.fold(player)
            player.history.append("fold")
            print('F')
            return "fold"



    def determine_winner(self):
        active_players = [p for p in self.players if p.in_hand]
        if not active_players:
            return None

        winner = max(active_players, key=lambda p: p.evaluate_hand(self.board))
        winner.money += self.pot
        self.pot = 0
        return winner

    def play_round(self):
        self.round_id += 1
        self.shuffle_deck()
        self.deal_cards()
        self.pot = 0
        self.current_bet = 0

        # Starting player bets $1
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

        # Determine the winner
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

# Example usage
player1 = Player("Alice", 10)
player2 = Player("Bob", 10)
game = GutsGame(player1,player2)
game.play_game()
