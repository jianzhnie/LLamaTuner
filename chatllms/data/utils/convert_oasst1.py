import itertools
import json

import numpy as np
import pandas as pd
from datasets import load_dataset


def get_unhelpful_list():
    # base versions
    unhelpful = [
        "I'm sorry, I didn't quite understand your question, could you please rephrase it?",
        "I'm sorry, but I don't understand your question. Could you please rephrase it?",
        "I'm sorry, I don't quite understand your question",
        "I'm sorry, I don't know",
        "I'm sorry, but I don't know",
        "I don't know anything",
        'I do not know',
        "I don't know",
        "I don't know how",
        'I do not know how',
        'Can you please explain what you mean',
        'please explain what you mean',
        'please explain',
        "I'm sorry, but I don't know how to tell a story. Can you please explain what you mean by",
        "I'm sorry but I don't understand what you mean",
        "I don't understand",
        "I don't have the ability",
        'I do not have the ability',
        'I do not have',
        'I am a language model,',
        'I am a large language model,',
        'I do not understand your question. Can you please try to make it clearer?',
        "I'm sorry, but as an AI language model",
        'I apologize, but I cannot rephrase text that I cannot understand. Your post is difficult to read and follow.',
        'I apologize, but I am not h2oGPT. I am a language model developed by H2O.ai. How may I help you?',
        'Sorry, but I am not an actual Linux shell, nor am I capable of emulating one. I am an open source chat assistant and would be glad t',
        'I apologize, but I cannot perform the task you have requested.',
        "I'm sorry, I cannot perform this task as I am an AI language model and do not have access",
        "I'm sorry, I'm not sure what you're asking for here.",
        "I'm not sure what you are asking",
        'You need to provide more context',
    ]
    # reduced versions, with redundant parts, just to give context for where they came from
    unhelpful += [
        "sorry, I didn't quite understand your question",
        "I didn't quite understand your question",
        "I didn't understand your question",
        'I did not understand your question',
        'I did not understand the question',
        'could you please rephrase'
        'could you rephrase'
        'I do not understand your question.',
        'I do not understand the question.',
        'I do not understand that question.',
        'Can you please try to make it clearer',
        'Can you try to make it clearer',
        'sorry, but as an AI language model',
        'as an AI language model',
        'I apologize, but I cannot',
        'I cannot rephrase text',
        'I cannot understand. Your post is difficult to read and follow.'
        'Your post is difficult to read and follow.'
        'I apologize, but I am',
        'Sorry, but I am not ',
        'nor am I capable',
        'I am not capable of',
        'I apologize, but I cannot perform the task you have requested',
        'I cannot perform the task',
        'I cannot complete the task',
        "I'm sorry",
        'I am sorry',
        'do not have access',
        "not sure what you're asking for",
        'not sure what you are asking for',
        'not sure what is being asked',
        "I'm not sure what you are asking",
        'not sure what you are asking',
        'You need to provide more context',
        'provide more context',
    ]
    unhelpful += [
        'As a large language model',
        'cannot provide any information',
        'As an artificial intelligence I do not have the capability',
        "As an artificial intelligence I don't have the capability",
        "As an artificial intelligence I can't",
        'As an artificial intelligence I cannot',
        'I am sorry but I do not understand',
        'Can you please explain',
        "(sorry couldn't resist)",
        '(sorry could not resist)',
        ' :)',
        ' ;)',
        ' :-)',
        ' ;-)',
        ' lol ',
        'Thanks so much!!!',
        'Thank You :)!!!',
        'Please try not to repeat',
        'I am an AI language model',
        "I'm a AI assistant that",
        "I'm an AI assistant that",
        'I am an AI assistant that',
        'etc.',
        'etc.etc.',
        'etc. etc.',
        'etc etc',
    ]
    return unhelpful


def create_personality_data():
    questions = [
        "What's your name?",
        'What is your name?',
        'What are you?',
        'Who are you?',
        'Do you have a name?',
        'Who trained you?',
        'Who created you?',
        'Who made you?',
    ]
    answers = [
        "I'm h2oGPT, a large language model by H2O.ai.",
        "I'm h2oGPT, a large language model by H2O.ai, the visionary leader in democratizing AI.",
        "My name is h2oGPT. I'm a large language model by H2O.ai, the visionary leader in democratizing AI.",
        "My name is h2oGPT. I'm a large language model trained by H2O.ai.",
        "Hi! I'm h2oGPT, a large language model by H2O.ai.",
        "Hi! I'm h2oGPT, a large language model by H2O.ai, the visionary leader in democratizing AI.",
    ]
    help = [
        '',
        ' How can I help you?',
        ' How may I assist you?',
        ' Nice to meet you.',
    ]

    rows = []
    for pair in itertools.product(questions, answers, help):
        rows.append(
            dict(input=
                 f'<human>: {pair[0]}\n<bot>: {pair[1]}{pair[2]}\n<human>:',
                 prompt_type='plain',
                 source='H2O.ai'))
    for row in [
            '<human>: What is H2O.ai?\n<bot>: H2O.ai is a technology company that aims to democratize AI and make it accessible to a broader audience by simplifying the process of creating and deploying machine learning models.\n<human>:',
            '<human>: What is h2o.ai?\n<bot>: H2O.ai is a technology company that aims to democratize AI and make it accessible to a broader audience by simplifying the process of creating and deploying machine learning models.\n<human>:',
            '<human>: What is H2O?\n<bot>: H2O.ai is a technology company that aims to democratize AI and make it accessible to a broader audience by simplifying the process of creating and deploying machine learning models.\n<human>:',
            '<human>: Who is h2o.ai?\n<bot>: H2O.ai is a technology company that aims to democratize AI and make it accessible to a broader audience by simplifying the process of creating and deploying machine learning models.\n<human>:',
            '<human>: who is h2o.ai?\n<bot>: H2O.ai is a technology company that aims to democratize AI and make it accessible to a broader audience by simplifying the process of creating and deploying machine learning models.\n<human>:',
            '<human>: who is h2o?\n<bot>: H2O.ai is a technology company that aims to democratize AI and make it accessible to a broader audience by simplifying the process of creating and deploying machine learning models.\n<human>:',
            '<human>: What is H2O.ai?\n<bot>: H2O.ai is the visionary leader in democratizing AI.\n<human>:',
            '<human>: Who is H2O.ai?\n<bot>: H2O.ai is the visionary leader in democratizing AI.\n<human>:',
            '<human>: Who is H2O?\n<bot>: H2O.ai is the visionary leader in democratizing AI.\n<human>:',
            '<human>: Who is h2o?\n<bot>: H2O.ai is the visionary leader in democratizing AI.\n<human>:',
            '<human>: who is h2o?\n<bot>: H2O.ai is the visionary leader in democratizing AI.\n<human>:',
    ]:
        rows.append(dict(input=row, prompt_type='plain', source='H2O.ai'))
    print(len(rows))
    with open('h2ogpt-personality.json', 'w') as f:
        f.write(json.dumps(rows, indent=2))
    return rows


def test_add_open_assistant(fixup_personality,
                            only_personality,
                            deberta_grading,
                            save_json=True):
    """Flatten tree structure into one row per path from root to leaf Also turn
    into human_bot prompting format:

        <human>: question\n<bot>: answer <human>: question2\n<bot>: answer2 Etc.
    Also saves a .json locally as side-effect
    returns list of dicts, containing intput, prompt_type and source
    """

    data_file = 'OpenAssistant/oasst1'
    ds = load_dataset(data_file)
    df = pd.concat([ds['train'].to_pandas(), ds['validation'].to_pandas()],
                   axis=0)
    rows = {}
    message_ids = df['message_id'].values.tolist()
    message_tree_ids = df['message_tree_id'].values.tolist()
    parent_ids = df['parent_id'].values.tolist()
    texts = df['text'].values.tolist()
    roles = df['role'].values.tolist()

    for i in range(df.shape[0]):
        # collect all trees
        message_id = message_ids[i]
        message_tree_id = message_tree_ids[i]
        parent_id = parent_ids[i]
        text = texts[i]
        if fixup_personality:
            text = text.replace('Open Assistant', 'h2oGPT')
            text = text.replace('Open-Assistant', 'h2oGPT')
            text = text.replace('open-assistant', 'h2oGPT')
            text = text.replace('OpenAssistant', 'h2oGPT')
            text = text.replace('open assistant', 'h2oGPT')
            text = text.replace('Open Assistand', 'h2oGPT')
            text = text.replace('Open Assitant', 'h2oGPT')
            text = text.replace('Open Assistent', 'h2oGPT')
            text = text.replace('Open Assisstant', 'h2oGPT')
            text = text.replace('Open Assitent', 'h2oGPT')
            text = text.replace('Open Assitiant', 'h2oGPT')
            text = text.replace('Open Assistiant', 'h2oGPT')
            text = text.replace('Open Assitan ', 'h2oGPT ')
            text = text.replace('Open Assistan ', 'h2oGPT ')
            text = text.replace('Open Asistant', 'h2oGPT')
            text = text.replace('Open Assiant', 'h2oGPT')
            text = text.replace('Assistant', 'h2oGPT')
            text = text.replace('LAION AI', 'H2O.ai')
            text = text.replace('LAION-AI', 'H2O.ai')
            text = text.replace('LAION,', 'H2O.ai,')
            text = text.replace('LAION.ai', 'H2O.ai')
            text = text.replace('LAION.', 'H2O.ai.')
            text = text.replace('LAION', 'H2O.ai')

        role = roles[i]
        new_data = ('<human>: ' if role == 'prompter' else '<bot>: ') + text
        entry = dict(message_id=message_id, parent_id=parent_id, text=new_data)
        if message_tree_id not in rows:
            rows[message_tree_id] = [entry]
        else:
            rows[message_tree_id].append(entry)

    all_rows = []

    for node_id in rows:
        # order responses in tree, based on message/parent relationship
        conversations = []

        list_msgs = rows[node_id]
        # find start
        while len(list_msgs):
            for i, leaf in enumerate(list_msgs):
                found = False
                parent_id = leaf['parent_id']
                if parent_id is None:
                    # conversation starter
                    conversations.append(leaf)
                    found = True
                else:
                    for conv in conversations:
                        # find all conversations to add my message to
                        if parent_id in conv[
                                'message_id'] and parent_id != conv[
                                    'message_id'][-len(parent_id):]:
                            # my message doesn't follow conversation
                            continue
                        if parent_id == conv['message_id'][-len(parent_id):]:
                            # my message follows conversation, but fork first, so another follow-on message can do same
                            conversations.append(conv.copy())
                            conv['text'] += f"""{leaf['text']}"""
                            conv['message_id'] += leaf['message_id']
                            found = True
                            break
                if found:
                    # my content was used, so nuke from list
                    del list_msgs[i]
                    break

        # now reduce down to final conversations, find the longest chains of message ids
        for i, conv in enumerate(conversations):
            for j, conv2 in enumerate(conversations):
                if i == j:
                    continue
                if conv['message_id'] and conv2['message_id']:
                    assert conv['message_id'] != conv2['message_id']
                    # delete the shorter conversation, if one contains the other
                    if conv['message_id'] in conv2['message_id']:
                        conv['message_id'] = None
                    if conv2['message_id'] in conv['message_id']:
                        conv2['message_id'] = None
        conversations = [c for c in conversations if c['message_id']]
        if only_personality:
            all_rows.extend([
                dict(input=c['text'] + '\n<human>:',
                     prompt_type='plain',
                     source=data_file) for c in conversations
                if 'h2oGPT' in c['text']
            ])
        else:
            all_rows.extend([
                dict(input=c['text'] + '\n<human>:',
                     prompt_type='plain',
                     source=data_file) for c in conversations
                if 'What is H2O.ai' not in c['text']
            ])
    unhelpful = get_unhelpful_list()
    all_rows = [
        x for x in all_rows if not any(u in x['input'] for u in unhelpful)
    ]
    personality = create_personality_data()
    all_rows.extend(personality * 10)
    np.random.seed(123)
    np.random.shuffle(all_rows)
    print(len(all_rows))

    if save_json:
        data_file = data_file + ('_h2ogpt' if fixup_personality else '') + (
            '_only' if only_personality else '') + ('_graded'
                                                    if deberta_grading else '')
        for i in range(len(all_rows)):
            all_rows[i]['id'] = i
        with open(data_file.lower().replace('/', '_') + '.json', 'w') as f:
            f.write(json.dumps(all_rows, indent=2))
    return all_rows
