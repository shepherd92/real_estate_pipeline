#!/usr/bin/env python
"""Async URL requests returning BeautifulSoups."""


from asyncio import run, gather
from asyncio.exceptions import TimeoutError as TOError
from typing import Optional
from time import sleep

import numpy as np
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from aiohttp import (
    ClientSession,
    ClientTimeout,
    TCPConnector
)
from aiohttp.client_exceptions import (
    ClientOSError,
    ClientResponseError,
    ServerDisconnectedError,
    ClientConnectorError
)


class Request:
    """Handle async requests for several URL-s."""

    def __init__(
        self,
        initial_num_of_parallel_requests: int,
        dynamically_adjust_parallel_requests: bool
    ) -> None:
        """Set initial values for Request class."""
        self.parallel_requests = initial_num_of_parallel_requests
        self.dynamically_adjust_parallel_requests = dynamically_adjust_parallel_requests
        self.upper_threshold = 5

        self.total_num_of_requested_urls = 0
        self.total_time_slept = 0

    def __call__(self, urls: list[str]) -> list[Optional[BeautifulSoup]]:
        """Interface function to request several URL-s in parallel."""
        if len(urls) == 0:
            return []

        results = run(_wrapper(urls, self.parallel_requests))
        results = [list(tuple) for tuple in results]

        if self.dynamically_adjust_parallel_requests:
            self.total_num_of_requested_urls += len(urls)
            attempts = np.array([result[1] for result in results])
            self.total_time_slept += sum(2 ** attempts[attempts > 1])

            if self.total_num_of_requested_urls >= 1000:
                # adjust parallel_requests if is necessary
                self._adjust_parallel_reqests()

        return [result[0] for result in results]  # return only the soups

    def _adjust_parallel_reqests(self) -> None:
        # evaluate requests based on waiting time

        mean_sleep_time = self.total_time_slept / self.total_num_of_requested_urls

        if mean_sleep_time > 0.1:
            self.parallel_requests -= 1
            self.upper_threshold = self.parallel_requests
            print(f'Parallel requests decreased to {self.parallel_requests}')
        elif np.isclose(mean_sleep_time, 0.0):
            self.parallel_requests += 1
            print(f'Parallel requests increased to {self.parallel_requests}')

        self.parallel_requests = np.clip(
            self.parallel_requests,
            1, self.upper_threshold
        )
        self.total_num_of_requested_urls = 0
        self.total_time_slept = 0


async def _wrapper(urls: list[str], parallel_requests: int) -> list[tuple[Optional[BeautifulSoup], int]]:
    """Wrap async URL requests with common TCP connector."""
    connector = TCPConnector(limit_per_host=parallel_requests)
    session = ClientSession(raise_for_status=True, connector=connector, timeout=ClientTimeout(total=600))

    results = await gather(*[_request_url(url, session) for url in urls])
    await session.close()
    return results


async def _request_url(url: str, session: ClientSession, attempt: int = 1) -> tuple[Optional[BeautifulSoup], int]:
    max_attempts = 16  # exit with timeout after 2**16 seconds (~18 hours)
    user_agent = UserAgent()

    try:
        header = {'User-Agent': user_agent.random}
        response = await session.get(url, headers=header)
        text = await response.text()
        soup = BeautifulSoup(text, 'lxml')
        return soup, attempt

    except (
        ClientResponseError,
        ServerDisconnectedError,
        ClientConnectorError,
        ClientOSError,
        TOError
    ) as exception:
        if attempt >= 5:
            print(f'Exception: {type(exception)} at attempt {attempt}; {exception}', flush=True)

        if isinstance(exception, ClientResponseError):
            if exception.status in (404, 410):
                # page not found
                return None, attempt

        if attempt <= max_attempts:
            # await sleep(2**attempt)
            sleep(2**attempt)  # synchronous on purpose
            return await _request_url(url, session, attempt=attempt+1)

        raise


if __name__ == '__main__':
    request = Request(initial_num_of_parallel_requests=3, dynamically_adjust_parallel_requests=True)
    request(['https://ingatlan.com/32384310'] * 100)
