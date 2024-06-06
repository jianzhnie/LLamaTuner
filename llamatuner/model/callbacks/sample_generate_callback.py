import argparse
from dataclasses import dataclass
from typing import Any, Dict

from transformers import PreTrainedTokenizer, TrainerCallback


@dataclass
class SampleGenerateCallback(TrainerCallback):
    """A callback that generates text samples from a pre-trained language model
    during training.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer used to preprocess inputs.
        max_new_tokens (int): The maximum number of tokens to generate in response to each input.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer,
                 generation_config: argparse.Namespace, logger: None):
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.logger = logger

        # Define input prompts to generate text from
        self.sample_inputs = [
            '用一句话描述地球为什么是独一无二的。',
            '中国是否应该推出刺激政策救楼市？',
            '如何更好地融入新工作圈子',
            '帮我把这段文字转换成鲁迅作品里的语气：昨天上午，算几个数学问题时越算越难受，有想要撕掉草稿纸的冲动思维也变得缓慢，见字忘意，感觉大脑里是一团浆糊，阻力很大。'
            '我怀疑自己抑郁又犯了，站起身离开了书桌。走出大门，开始跑步，运动，希望能借此缓解。我不想再吃药，我担心不吃药是否能恢复。稍微运动后，大吃了一顿，路上不停的对自己说，我可以.',
            '回来后，感觉似乎确实好一些。',
            '给我写一篇大模型的新闻稿',
            '你觉得人类哪些工作岗位会被AI替代？',
            '请帮我写一封中式婚礼请帖，用于邀请亲朋好友参加我的婚礼！',
            '帮我写一篇八百字以上的作文，主题是：当代青年面对时代的挑战如何肩负起民族复兴的伟大任务',
            '请仿照李荣浩的风格写一首表现爱情的歌曲，以“辣椒酱”为题。',
            '秦王朝时期十大将军是？其主要功绩是什么？',
            '帮我写一段广告，关于房产销售的，我们的房子首付低，赠送面积大，还免两年物业费！',
            '请帮我设计一个时长为3天的北京旅游行程，行程的内容不要太紧凑，使用地铁作为交通工具，并前往前门、天安门、天坛公园、鸟巢游览，同时预留一天的时间游玩环球影城。',
            '一个笼子里面有若干只鸡和兔子，总共有50只脚和18个头，求鸡和兔子各有多少只？',
            '生成一篇短篇小说，故事情节为一个年轻人在旅途中遇到了一位神秘的老人，老人告诉他一个令人意想不到的秘密，最终年轻人的生活因此发生了翻天覆地的变化。',
            '导师想要我论文的一作，我应该怎么办？',
            '我现在很无聊，可以讲点有趣的事情吗？',
            '一项工程，甲、乙两队合作20天完成，乙丙两队合作60天完成，丙丁两队合作30完成，甲丁合作多少天完成?',
            '如果一位孕妇走上了公交车，但是车上没有空位了。请模拟一位热心乘客给孕妇让座的对话。',
            '桃花潭水深千尺，不及汪伦送我情。体现的是怎样的心情？',
            '编写一个简单的自动化脚本，用于批量操作文件或目录。脚本功能可以自由选择，如复制、压缩、重命名、删除等。脚本语言可使用Python、Shell、Perl等，代码长度不少于100行。',
            '音乐可以洗涤人的灵魂吗？',
        ]

    def on_evaluate(self, args: Any, state: Dict[str, Any], control: Any,
                    **kwargs: Any) -> None:
        """Generates text samples from the language model during evaluation.

        Args:
            args (Any): Trainer arguments, not used in this method.
            state (Dict[str, Any]): Trainer state dictionary, not used in this method.
            control (Any): Trainer control object, not used in this method.
            kwargs (Dict[str, Any]): Keyword arguments passed to the method, including the pre-trained
                language model (under the key 'model') and any additional parameters needed for generation.

        Returns:
            None
        """
        logger = self.logger
        logger.info('Generating sample text during evaluation...')

        # Check if the pre-trained language model is available
        if 'model' in kwargs:
            model = kwargs['model']

            # Generate text for each input prompt
            for instruction in self.sample_inputs:
                # Preprocess input prompt and convert to tensor
                inputs = f'{instruction}\n\n### Response: '
                inputs = self.tokenizer(inputs, return_tensors='pt')
                inputs = inputs.to(model.device)

                # Generate text from input prompt
                generation_output = model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                )

                # Decode generated text and log it
                generated_text = self.tokenizer.decode(generation_output[0])
                logger.info(f'Input prompt: {instruction}')
                logger.info(f'Generated text: {generated_text}')

        else:
            logger.info(
                'Pre-trained language model not found in kwargs, skipping.')
