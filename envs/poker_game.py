from random import shuffle

class PokerGame(object):
    """
    Poker Game
    三个人一起玩扑克游戏，每次从1到10中随机抽一张作为奖励牌，
    三个人没人出一张牌比大小，数字大的人获胜，获得奖励牌对应的点数的积分，
    如果三个人的出牌相同，则所有人都失败，得分为0。
    如果两个人出牌相同，则这两人失败，另一个人获得这轮的奖励牌，
    游戏共进行10轮，最终得分最高的人获胜。
    """
    
    def __init__(self) -> None:
        self._reset_game()
    def _reset_game(self):
        self.cards_a = list(range(1, 11))
        self.cards_b = list(range(1, 11))
        self.cards_c = list(range(1, 11))
        self.cards_r = list(range(1, 11))

        self.current_reward = None
        self.current_result_a = None
        self.current_result_b = None
        self.current_result_c = None

        self.total_reward_a = 0
        self.total_reward_b = 0
        self.total_reward_c = 0

    def _set_current_result(self, card_a, card_b, card_c):
        self.current_result_a = card_a
        self.current_result_b = card_b
        self.current_result_c = card_c
        self.total_reward_a += self.current_result_a
        self.total_reward_b += self.current_result_b
        self.total_reward_c += self.current_result_c

    def _bet_core(self, card_a, card_b, card_c):
        assert self.current_reward
        if card_a == card_b == card_c:
            self._set_current_result(0, 0, 0)
        elif card_a == card_b or card_b == card_c or card_a == card_c:
            if card_a == card_b:
                self._set_current_result(0, 0, self.current_reward)
            elif card_b == card_c:
                self._set_current_result(self.current_reward, 0, 0)
            else:
                self._set_current_result(0, self.current_reward, 0)
        else:
            max_card = max(card_a, card_b, card_c)
            if max_card == card_a:
                self._set_current_result(self.current_reward, 0, 0)
            elif max_card == card_b:
                self._set_current_result(0, self.current_reward, 0)
            else:
                self._set_current_result(0, 0, self.current_reward)
        self.current_reward = None

    def reset(self):
        self._reset_game()

    @property
    def terminal(self) -> bool:
        return len(self.cards_r) == 0

    def lottery(self) -> int:
        assert self.cards_r
        shuffle(self.cards_r)
        self.current_reward = self.cards_r.pop()
        return self.current_reward

    @property
    def total_reward(self) -> tuple:
        return self.total_reward_a, self.total_reward_b, self.total_reward_c
    
    @property
    def winner(self):
        max_total_reward = max(self.total_reward_a, self.total_reward_b, self.total_reward_c)
        return max_total_reward == self.total_reward_a, max_total_reward == self.total_reward_b, max_total_reward == self.total_reward_c

    def bet(self, card_a:int=None, card_b:int=None, card_c:int=None):
        """如果没有提供卡牌，或者提供无效卡牌，则从剩余牌中随机抽一张牌"""
        if card_a is None or card_a not in self.cards_a:
            shuffle(self.cards_a)
            card_a = self.cards_a.pop()
        else:
            self.cards_a.remove(card_a)
        
        if card_b is None or card_b not in self.cards_b:
            shuffle(self.cards_b)
            card_b = self.cards_b.pop()
        else:
            self.cards_b.remove(card_b)

        if card_c is None or card_c not in self.cards_c:
            shuffle(self.cards_c)
            card_c = self.cards_c.pop()
        else:
            self.cards_c.remove(card_c)

        self._bet_core(card_a, card_b, card_c)
        return self.current_result_a, self.current_result_b, self.current_result_c


class PokerGameEnv(object):
    pass


if __name__ == '__main__':

    game = PokerGame()
    scores = []
    winners = []
    for _ in range(10000):
        game.reset()
        cards_a = list(range(1, 11))
        cards_b = list(range(10, 0, -1))
        while not game.terminal:
            card = game.lottery()
            game.bet(card_a=cards_a.pop(), card_b=cards_b.pop(),card_c=card)
        scores.append(game.total_reward)
        winners.append(game.winner)

    scores = list(zip(*scores))
    winners = list(zip(*winners))

    gamers = ["A", "B", "C"]

    for gamer, score in zip(gamers, scores):
        print(f"Player {gamer}, avverage reward: {sum(score)/len(score):.2f}")

    for gamer, winner in zip(gamers, winners):
        print(f"Player {gamer}, win rate: {sum(winner)/len(winner)*100:.2f}%")

