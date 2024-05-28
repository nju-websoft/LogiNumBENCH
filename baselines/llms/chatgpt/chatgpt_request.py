import json
import os
import time
import queue
import re
import threading

from typing import List, Dict

import openai
from openai.error import RateLimitError
from tqdm import tqdm


lock = threading.Lock()


def get_prompt(data):
    return data['prompt'], data['instr']


class QueueWithBar(queue.Queue):
    def __init__(self, total):
        super().__init__()
        self.tqdm = iter(tqdm(range(total), total=total, ncols=75))

    def get_with_bar(self):
        try:
            next(self.tqdm)
        except StopIteration:
            pass
        return self.get(block=False)


def chatgpt_request(api_key, prompt, instr):
    success = False
    error_count = 0
    while not success:
        try:
            # with lock:
            openai.api_key = api_key

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # "gpt-3.5-turbo",
                # model="text-davinci-003",
                messages=[
                    {"role": "system", "content": instr},
                    {"role": "user", "content": prompt},
                ],
            )
            success = True
            if error_count > 5:
                raise NotImplementedError
        except RateLimitError as e:
            if 'You exceeded your current quota, please check your plan and billing details.' in str(e):
                print(f'no money api key 555: {api_key}')
                raise NotImplementedError
            print(f'RateLimitError: {e}')
            error_count += 1
            time.sleep(3)
        except Exception as e:
            error_count += 1
            print(f'Warning! waiting for try again: {e}')
            time.sleep(3)

    return response


def data_dump(data_path, response_data):
    with lock:
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as input_file:
                existing_data = json.load(input_file)
        else:
            existing_data = {}

        # 更新已有数据
        existing_data.update(response_data)

        # 将更新后的数据写回文件
        with open(data_path, 'w', encoding='utf-8') as output_file:
            json.dump(existing_data, output_file, ensure_ascii=False, indent=4)


class ChatGPTThread(threading.Thread):
    def __init__(self, queue, api_key, output_data_path, response_data):
        threading.Thread.__init__(self)
        self.queue = queue
        self.api_key = api_key

        self.data_path = output_data_path
        self.response_data = response_data

    def run(self):
        count = 0
        while True:
            try:
                data = self.queue.get_with_bar()
                key = data['id']
                if key in self.response_data:
                    continue
                prompt, instr = get_prompt(data)
                try:
                    raw_response = chatgpt_request(
                        self.api_key, prompt=prompt, instr=instr)
                    # print(raw_response)
                except Exception as e:
                    self.queue.put(data)
                    print(f'exception: {e}')
                    break
                except NotImplementedError:
                    self.queue.put(data)
                    print(f'thead exit due to too many errors')
                    break
                self.response_data[key] = data.copy()
                self.response_data[key]['raw_response'] = raw_response
                self.response_data[key]['prompt'] = prompt

                count += 1
                if count % 10 == 0:
                    data_dump(self.data_path, self.response_data)
                time.sleep(3)
            except queue.Empty:  # 队列为空时跳出循环
                data_dump(self.data_path, self.response_data)
                break


class ChatGPTThreadPool:
    def __init__(self, tasks, thread_num: int, api_keys: List[str], output_data_path):
        assert thread_num == len(api_keys)
        self.thread_pool = []
        self.tasks = tasks
        response_data = {} if not os.path.exists(output_data_path) else json.load(
            open(output_data_path, 'r', encoding='utf-8'))
        for i in range(thread_num):
            self.thread_pool.append(ChatGPTThread(
                tasks, api_keys[i], output_data_path, response_data))
        self.response_data = response_data
        self.output_data_path = output_data_path

    def run(self):
        for t in self.thread_pool:
            t.start()
        for t in self.thread_pool:
            t.join()
        # self.tasks.join()
        data_dump(self.output_data_path, self.response_data)
        print('All done!')
