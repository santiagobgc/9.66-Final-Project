#Model 2
import random
import treys
import pymc as pm
from treys import Deck
from treys import Card


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
        for opponent_hand in  pos_opponent_hands:
            opponent_strength = evaluator.evaluate(self.board, list(opponent_hand))
            if hand_strength < opponent_strength:
                wins += 1
            elif hand_strength == opponent_strength:
                ties += 1
            else:
                losses += 1

        total_scenarios = len(pos_opponent_hands)
        win_prob = wins / total_scenarios
        tie_prob = ties / total_scenarios
        loss_prob = losses / total_scenarios
        return (win_prob, tie_prob, loss_prob)

    def rotate_starting_player(self):
        self.current_player_index = (self.current_player_index + 1) % len(self.players)


    def player_action(self, player):
        if not player.in_hand or player.money <= 0:
            self.fold(player)
        win_prob, tie_prob, lose_prob = self.win_prob(player)
        fold_ev = 0
        call_ev = (self.pot) * win_prob - (self.current_bet - player.total_bet) * lose_prob
        raise_ev = (self.pot + self.current_bet) * win_prob - (2 * self.current_bet - player.total_bet) * lose_prob
        if player.name == "Jack":
          jack_fold = (jack_samples < 0.4 - win_prob).mean()
          jack_call = ((0.4 - win_prob < jack_samples) & (jack_samples < 0.6 - win_prob)).mean()
          jack_raise = (jack_samples > 0.6 - win_prob).mean()
          values = [jack_fold, jack_call, jack_raise]
          normal_draw = pm.draw(jack_normal)
        else:
          santi_fold = (santi_samples < 0.4 - win_prob).mean()
          santi_call = ((0.4 - win_prob < santi_samples) & (santi_samples < 0.6 - win_prob)).mean()
          santi_raise = (santi_samples > 0.6 - win_prob).mean()
          values = [santi_fold, santi_call, santi_raise]
          normal_draw = pm.draw(santi_normal)
        santi_greater = (santi_samples > 0.1).mean()
        labels = ["Fold", "Call", "Raise"]
        plt.bar(labels, values, color=["blue", "orange", "red"])
        plt.ylabel("Proportion")
        if player.name == "Jack":
          plt.title("Jack Probability of each Action")
        else:
          plt.title("Santi Probability of each Action")
        plt.ylim(0, 1)
        plt.show()
        #print(fold_ev,call_ev, raise_ev)
        win_prob += normal_draw
        if win_prob < 0.4:
            self.fold(player)
            return "fold"
        elif win_prob < 0.6:
            if player.money >= self.current_bet - player.total_bet:
                 self.call(player)
                 return "call"
            else:
                 self.fold(player)
                 return "fold"

        else:
            if player.money >= 2 * self.current_bet - player.total_bet:
                self.raise_bet(player, current_bet)
                return "raise"
            elif player.money >= self.current_bet - player.total_bet:
                self.call(player)
                return "call"
            else:
                self.fold(player)
                return "fold"


    def determine_winner(self):
        active_players = [p for p in self.players if p.in_hand]
        if not active_players:
            return None

        winner = min(active_players, key=lambda p: p.evaluate_hand(self.board))
        winner.money += self.pot
        self.pot = 0
        print(winner)
        return winner

    def play_round(self, number):
        self.round_id += 1
        self.shuffle_deck()
        self.real_game(boards[number], santi_hands[number], jack_hands[number])
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
            for i in range(9):
              winner = self.play_round(i)
            if winner:
                print(f"{winner.name} wins the pot! Current money: {winner.money}")

        winner = max(self.players, key=lambda p: p.money)
        print(f"Game over! The winner is {winner.name} with ${winner.money}")

player1 = Player("Santi", 10)
player2 = Player("Jack", 10)
game = GutsGame(player1,player2)
for i in range(7):
  game.player1.money = santi_money[i]
  game.player2.money = jack_money[i]
  game.play_round(i)
game = GutsGame(player1,player2)
for i in range(7, 32):
  game.player1.money = santi_money[i]
  game.player2.money = jack_money[i]
  game.play_round(i)


