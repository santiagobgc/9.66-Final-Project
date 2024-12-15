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
import matplotlib
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
        self.in_hand = True
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
        #Card.print_pretty_cards(self.board)
    
    def real_game(self, board, player1hand, player2hand):
        self.board = board
        for player in self.players:
            player.reset()
        self.player1.cards = player1hand
        self.player2.cards = player2hand
        #Card.print_pretty_cards(self.player2.cards)
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
        if player.name == "Jack":
            opponent_fold_if_raise = (0.5 - raise_count * data.raise_samples  + data.santi_samples + (1 + player.history.count("raise")) * data.raise_samples < 0.4).mean()
        else:
            opponent_fold_if_raise = (0.5 - raise_count * data.raise_samples + data.jack_samples + (1 + player.history.count("raise")) * data.raise_samples < 0.4).mean()
        self_bluff = (0.2/(player.history.count("Raise") + 1))
        if player.name == "Jack":
          jack_fold = (data.jack_samples < 0.4 - win_prob - raise_count * data.raise_samples).mean()
          jack_fold_with_bluff = jack_fold * (1-self_bluff)
          jack_call = ((0.4 - win_prob - raise_count * data.raise_samples < data.jack_samples) 
                       & (data.jack_samples < 0.6 - win_prob- raise_count * data.raise_samples)).mean()
          jack_raise = (data.jack_samples > 0.6 - win_prob- raise_count * data.raise_samples).mean()
          jack_raise_with_bluff = self_bluff * jack_fold + jack_raise
          values = [jack_fold_with_bluff, jack_call, jack_raise_with_bluff]
          normal_draw = pm.draw(data.jack_normal)
        else:
          santi_fold = (data.santi_samples < 0.4 - win_prob - raise_count * data.raise_samples).mean()
          santi_fold_with_bluff = santi_fold * (1-self_bluff)
          santi_call = ((0.4 - win_prob - raise_count * data.raise_samples < data.santi_samples) 
                        & (data.santi_samples < 0.6 - win_prob - raise_count * data.raise_samples)).mean()
          santi_raise = (data.santi_samples > 0.6 - win_prob - raise_count * data.raise_samples).mean()
          santi_raise_with_bluff = self_bluff * santi_fold + santi_raise
          values = [santi_fold_with_bluff, santi_call, santi_raise_with_bluff]
          normal_draw = pm.draw(data.santi_normal)
        return values, opponent_fold_if_raise, normal_draw
        


    def player_action(self, player):
        if not player.in_hand or player.money <= 0:
            self.fold(player)
            return "fold"
        for opponent in self.players:
            if opponent != player:
                raise_count = opponent.history.count("raise")
        win_prob, tie_prob, lose_prob, all_hands, all_probabilities = self.win_prob(player)
        values, opponent_fold_if_raise, normal_draw = self.get_probability_dist(player, win_prob, raise_count)
        #print(player.name, values)
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
        if win_prob > 0.6: 
            if player.name == "Santi":
                self.call(player)
                player.history.append("call")
                print(player.name, 'call')
                return "call"
            if player.money >= 2 * self.current_bet - player.total_bet :
                self.raise_bet(player, self.current_bet)
                player.history.append("raise")
                print(player.name, "raise")
                return "raise"
        elif win_prob > 0.4:
            if player.money >= self.current_bet - player.total_bet:
                self.call(player)
                player.history.append("call")
                print(player.name, 'call')
                return "call"
        else:
            if random.random() < 0.2/(1 + player.history.count("Raise")) and player.money >= 2 * self.current_bet - player.total_bet:
                self.raise_bet(player, self.current_bet)
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
        self.deal_cards()
        #self.real_game(data.boards[number], data.santi_hands[number], data.jack_hands[number])
        self.pot = 0
        self.current_bet = 0

        actions = []
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
            actions.append(action)
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

        all_actions.append(actions)
        winner = self.determine_winner()
        #print(f"Winner of this round: {winner.name} with ${winner.money}")
        self.rotate_starting_player()
        return winner

    def play_game(self):
        while all(p.money > 0 for p in self.players):
            #print(f"Starting a new round. Pot: ${self.pot}")
            winner = self.play_round(1)
            if winner:
                pass
                #print(f"{winner.name} wins the pot! Current money: {winner.money}")

        winner = max(self.players, key=lambda p: p.money)
        #print(f"Game over! The winner is {winner.name} with ${winner.money}")


all_actions = []
player1 = Player("Santi", 1000)
player2 = Player("Jack", 1000)
game = GutsGame(player1, player2)
game.play_game()
#print(all_actions.count(["raise",'call']) / len(all_actions))


'''
for i in range(7):
  game.player1.money = data.santi_money[i]
  game.player2.money = data.jack_money[i]
  print("Round", str(i+1))
  game.play_round(i)
  print("")
game = GutsGame(player1,player2)
for i in range(7, 32):
  game.player1.money = data.santi_money[i]
  game.player2.money = data.jack_money[i]
  print("Round", str(i+1))
  game.play_round(i)
  print('')
  '''
