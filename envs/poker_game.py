from random import shuffle
import numpy as np

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
        self.cards_a = list(range(10))
        self.cards_b = list(range(10))
        self.cards_c = list(range(10))
        self.cards_r = list(range(10))

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
        return len(self.cards_a) == 0

    def lottery(self) -> int:
        assert self.cards_r
        shuffle(self.cards_r)
        self.current_reward = self.cards_r.pop() + 1
        return self.current_reward

    @property
    def total_reward(self) -> tuple:
        return self.total_reward_a, self.total_reward_b, self.total_reward_c
    
    @property
    def winner(self):
        if not self.terminal:
            return False, False, False
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
    
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.poker_game = PokerGame()

    @property
    def _state(self) -> np.ndarray:
        current_reward_vector = [0] * 10
        if self.poker_game.current_reward is not None:
            current_reward_index = self.poker_game.current_reward - 1 
            current_reward_vector[current_reward_index] = 1

        remaining_reward_vector = [0] * 10
        for i in self.poker_game.cards_r:
            remaining_reward_vector[i] = 1
        
        remaining_cards_a = [0] * 10
        for i in self.poker_game.cards_a:
            remaining_cards_a[i] = 1

        remaining_cards_b = [0] * 10
        for i in self.poker_game.cards_b:
            remaining_cards_b[i] = 1
        
        remaining_cards_c = [0] * 10
        for i in self.poker_game.cards_c:
            remaining_cards_c[i] = 1

        if self.kwargs.get('state_ndim', 1) == 2:
            return np.array([current_reward_vector, remaining_reward_vector, remaining_cards_a, remaining_cards_b, remaining_cards_c])
        else:
            return np.array(current_reward_vector + remaining_reward_vector + remaining_cards_a + remaining_cards_b + remaining_cards_c)

    @property
    def state_dim(self) -> int:
        if self.kwargs.get('state_ndim', 1) == 2:
            return (5, 10)
        else:
            return 50
        
    @property
    def action_dim(self) -> int:
        return 10
    
    @property
    def action_mask(self) -> list:
        mask = [0] * 10
        for i in self.poker_game.cards_a:
            mask[i] = 1
        return mask

    def reset(self):
        self.poker_game.reset()
        self.poker_game.lottery()
        obs = self._state
        info = {
            "total_reward": self.poker_game.total_reward,
            "winner": self.poker_game.winner
        }
        return obs, info
        

    def sample_action(self) -> int:
        shuffle(self.poker_game.cards_a)
        return self.poker_game.cards_a[-1]

    def step(self, action):
        self.poker_game.bet(card_a=action)
        terminal = self.poker_game.terminal
        if not terminal:
            self.poker_game.lottery()
        reward = self.poker_game.current_result_a
        obs = self._state
        done = self.poker_game.winner[0]
        
        info = {
            "total_reward": self.poker_game.total_reward,
            "winner": self.poker_game.winner
        }

        return obs, reward, done, terminal, info

def poker_game_raw() -> PokerGameEnv:
    return PokerGameEnv(state_ndim=1)

def poker_game_raw_2d() -> PokerGameEnv:
    return PokerGameEnv(state_ndim=2)

def test_poker_game():
    game = PokerGame()
    scores = []
    winners = []
    for _ in range(10000):
        game.reset()
        cards_a = list(range(10))
        cards_b = list(reversed(range(10)))
        while not game.terminal:
            card = game.lottery()
            game.bet(card_a=cards_a.pop(), card_b=cards_b.pop())
        scores.append(game.total_reward)
        winners.append(game.winner)

    scores = list(zip(*scores))
    winners = list(zip(*winners))

    gamers = ["A", "B", "C"]

    for gamer, score in zip(gamers, scores):
        print(f"Player {gamer}, avverage reward: {sum(score)/len(score):.2f}")

    for gamer, winner in zip(gamers, winners):
        print(f"Player {gamer}, win rate: {sum(winner)/len(winner)*100:.2f}%")


def test_poker_game_env():
    env = PokerGameEnv(state_ndim=1)
    obs, info = env.reset()
    print(obs)
    print(info)

    while True:
        action = env.sample_action()
        print(f"{action=}")
        obs, reward, done, terminal, info = env.step(action)
        print(f"{reward=}")
        print(info)
        print(obs)
        
        if done or terminal:
            break

if __name__ == '__main__':
    # test_poker_game()
    test_poker_game_env()


