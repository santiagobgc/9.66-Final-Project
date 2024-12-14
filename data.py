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
raise_winprob_change = [-0.24, 0.1, -0.125, -0.05, 0.05, -0.05, -0.2 -0.1, -0.1, 0, -0.2, -0.1, -0.05, -0.1]
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

def parse_cards(card_string):
    return [Card.new(card.strip()[0]+card.strip()[1].lower()) for card in card_string.split(',')]

for board, hand1, hand2 in game2:
    boards.append(parse_cards(board))
    santi_hands.append(parse_cards(hand1))
    jack_hands.append(parse_cards(hand2))

def win_prob(player_name):
  probs = []
  evaluator = treys.Evaluator()
  for i in range(len(boards)):
    test_deck = treys.Deck()
    for card in boards[i]:
      test_deck.cards.remove(card)
    if player_name == "Jack":
      hand_strength = evaluator.evaluate(boards[i], jack_hands[i])
      for card in jack_hands[i]:
        test_deck.cards.remove(card)
    else:
      hand_strength = evaluator.evaluate(boards[i], santi_hands[i])
      for card in santi_hands[i]:
        test_deck.cards.remove(card)
    pos_opponent_hands = list(combinations(test_deck.cards, 2))
    wins = 0
    ties = 0
    losses = 0
    for opponent_hand in  pos_opponent_hands:
        opponent_strength = evaluator.evaluate(boards[i], list(opponent_hand))
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
    probs.append(win_prob)
  return probs


real_jack_winprobs = win_prob("Jack")
result_jack = [x - y for x, y in zip(jack_winprobs, real_jack_winprobs)]
jack_mean = sum(result_jack)/32
jack_var = sum((jack_mean - i)**2 for i in result_jack)/32
print(jack_mean, jack_var)
jack_sd = jack_var**(0.5)

real_santi_winprobs = win_prob("Santi")
result_santi = [x - y for x, y in zip(santi_winprobs, real_santi_winprobs)]
santi_mean = sum(result_santi)/32
santi_var = sum((santi_mean - i)**2 for i in result_santi)/32
print(santi_mean, santi_var)
santi_sd = santi_var**(0.5)

raise_winprob_change_mean = sum(raise_winprob_change)/len(raise_winprob_change)
raise_winprob_change_var = sum((raise_winprob_change_mean - i)**2 for i in raise_winprob_change)/len(raise_winprob_change)
raise_winprob_change_sd = raise_winprob_change_var**(0.5)


with pm.Model() as model:
    jack_normal = pm.Normal("jack_normal", mu=jack_mean, sigma=jack_sd)
    santi_normal = pm.Normal("santi_normal", mu=santi_mean, sigma=santi_sd)
    raise_normal = pm.Normal("raise_normal", mu=raise_winprob_change_mean, sigma=raise_winprob_change_sd)
    trace = pm.sample(2000, return_inferencedata=True, cores=2)

jack_samples = trace.posterior["jack_normal"].values.flatten()
santi_samples = trace.posterior["santi_normal"].values.flatten()
raise_samples = trace.posterior["raise_normal"].values.flatten()

'''
normal_draw = pm.draw(jack_normal, 100)
print(normal_draw)
'''

'''
x = np.linspace(-0.5, 0.5, 1000)
pdf = st.norm.pdf(x, santi_mean, santi_sd)
pdf2 = st.norm.pdf(x, jack_mean, jack_sd)
plt.plot(x, pdf, label="Santi", color="red")
plt.plot(x, pdf2, label="Jack", color="blue")
plt.show()
plt.show()
'''
