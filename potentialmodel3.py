#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:44:46 2024

@author: jackmarionsims
"""

import pymc as pm
import data
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
import treys
from treys import Deck
from treys import Card
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
    
    def real_game(self, board, player1hand, player2hand):
        self.board = board
        for player in self.players:
            player.reset()
        self.player1.cards = player1hand
        self.player2.cards = player2hand
        all_used = board + player1hand + player2hand
        for card in all_used:
            self.deck.cards.remove(card)
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
    
    def get_probability_dist(self, player, win_prob, raise_count):
        bluff_prob = 0.2/(raise_count + 1)
        fold_call_boundary = max((-bluff_prob*self.pot+self.current_bet-player.total_bet) / (self.pot +self.current_bet - player.total_bet), 0.2)
        if player.name == "Jack":
            opponent_fold_if_raise = (0.5 - raise_count * data.raise_samples  + data.santi_samples + (1 + player.history.count("raise")) * data.raise_samples < 0.4).mean()
        else:
            opponent_fold_if_raise = (0.5 - raise_count * data.raise_samples + data.jack_samples + (1 + player.history.count("raise")) * data.raise_samples < 0.4).mean()
        numerator = (1-(1-bluff_prob)*opponent_fold_if_raise)*(2*self.current_bet-player.total_bet) - (self.current_bet - player.total_bet)
        denominator = (1-(1-bluff_prob)*opponent_fold_if_raise)*(self.pot + 3 * self.current_bet - player.total_bet) - self.pot - (self.current_bet - player.total_bet)
        call_raise_boundary = min(numerator / denominator, 0.8)
        if player.name == "Jack":
          jack_fold = (data.jack_samples < fold_call_boundary - win_prob - raise_count * data.raise_samples).mean()
          jack_call = ((fold_call_boundary- win_prob - raise_count * data.raise_samples < data.jack_samples) 
                       & (data.jack_samples < call_raise_boundary - win_prob- raise_count * data.raise_samples)).mean()
          jack_raise = (data.jack_samples > call_raise_boundary - win_prob- raise_count * data.raise_samples).mean()
          values = [jack_fold, jack_call, jack_raise]
          normal_draw = pm.draw(data.jack_normal)
        else:
          santi_fold = (data.santi_samples < fold_call_boundary - win_prob - raise_count * data.raise_samples).mean()
          santi_call = ((fold_call_boundary - win_prob - raise_count * data.raise_samples < data.santi_samples) 
                        & (data.santi_samples < call_raise_boundary - win_prob - raise_count * data.raise_samples)).mean()
          santi_raise = (data.santi_samples > call_raise_boundary - win_prob - raise_count * data.raise_samples).mean()
          values = [santi_fold, santi_call, santi_raise]
          normal_draw = pm.draw(data.santi_normal)
        return values, opponent_fold_if_raise, normal_draw
        



    def player_action(self, player):
        if not player.in_hand or player.money <= 0:
            self.fold(player)
            return "fold"
        for opponent in self.players:
            opp = opponent
            if opponent != player:
                raise_count = opponent.history.count("raise")
                call_count = opponent.history.count("call")
        win_prob, tie_prob, lose_prob, all_hands, all_probabilities = self.win_prob(player)
        values, opponent_fold_if_raise, normal_draw = self.get_probability_dist(player, win_prob, raise_count)
        labels = ["Fold", "Call", "Raise"]
        plt.bar(labels, values, color=["blue", "orange", "red"])
        plt.ylabel("Proportion")
        if player.name == "Jack":
          plt.title("Jack Probability of each Action")
        else:
          plt.title("Santi Probability of each Action")
        plt.ylim(0, 1)
        #plt.show()
        win_prob += normal_draw
        win_prob = max(win_prob, 0)
        win_prob = min(win_prob, 1)
        bluff_prob = 0.2/(raise_count+1)
        fold_ev = 0 - bluff_prob * self.pot
        call_ev = (self.pot) * win_prob - (self.current_bet - player.total_bet) * (1-win_prob)
        raise_ev = (1 - (1-bluff_prob) * opponent_fold_if_raise) * ((self.pot + self.current_bet) * win_prob - (2 * self.current_bet - player.total_bet) * (1-win_prob))
        #print(f"Winning Probability: {win_prob}, Tie Probability: {tie_prob}, Losing Probability: {lose_prob}")
        #print(f"Raise: {raise_ev}, Call: {call_ev}, Fold: {fold_ev}")
        maximum = max(fold_ev, call_ev, raise_ev)
        if maximum == raise_ev: 
            if player.money >= self.current_bet + 1:
                self.raise_bet(player, 1)
                player.history.append("raise")
                print(player.name, "raise")
                return "raise"
        elif maximum == call_ev:  # Example threshold for calling
            if player.money >= self.current_bet:
                self.call(player)
                player.history.append("call")
                print(player.name, 'call')
                return "call"
        else:  # Fold if probabilities are low
            if random.random() < 0.2/(1 + player.history.count("Raise")) and player.money >= self.current_bet + 1:
                self.raise_bet(player, 1)
                player.history.append("raise")
                print(player.name, "raise")
                return "raise"
            else:
                self.fold(player)
                player.history.append("fold")
                print(player.name, 'fold')
                return "fold"



    def determine_winner(self):
        active_players = [p for p in self.players if p.in_hand]
        if not active_players:
            return None

        winner = min(active_players, key=lambda p: p.evaluate_hand(self.board))
        winner.money += self.pot
        self.pot = 0
        return winner

    def play_round(self, number):
        self.round_id += 1
        self.shuffle_deck()
        self.real_game(data.boards[number], data.santi_hands[number], data.jack_hands[number])
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

player1 = Player("Santi", 10)
player2 = Player("Jack", 10)
game = GutsGame(player1, player2)
for i in range(7):
  game.player1.money = data.santi_money[i]
  game.player2.money = data.jack_money[i]
  game.play_round(i)
game = GutsGame(player1,player2)
for i in range(7, 32):
  game.player1.money = data.santi_money[i]
  game.player2.money = data.jack_money[i]
  game.play_round(i)
