# syntax=docker/dockerfile:1
# Monorepo production image: Rails API + pinned Python classifier runtime + model/.
# Build from repository root:
#   docker build -t counterai-web .

ARG RUBY_VERSION=3.4.7
FROM docker.io/library/ruby:${RUBY_VERSION}-slim AS base

WORKDIR /rails

RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y curl libjemalloc2 libvips libpq5 && \
    ln -sf /usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2 /usr/local/lib/libjemalloc.so && \
    rm -rf /var/lib/apt/lists /var/cache/apt/archives

ENV RAILS_ENV="production" \
    BUNDLE_DEPLOYMENT="1" \
    BUNDLE_PATH="/usr/local/bundle" \
    BUNDLE_WITHOUT="development" \
    LD_PRELOAD="/usr/local/lib/libjemalloc.so"

FROM base AS build

RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y build-essential git libyaml-dev libpq-dev pkg-config && \
    rm -rf /var/lib/apt/lists /var/cache/apt/archives

WORKDIR /rails

COPY backend/Gemfile backend/Gemfile.lock backend/vendor ./

RUN bundle install && \
    rm -rf ~/.bundle/ "${BUNDLE_PATH}"/ruby/*/cache "${BUNDLE_PATH}"/ruby/*/bundler/gems/*/.git && \
    bundle exec bootsnap precompile -j 1 --gemfile

COPY backend/ .

RUN bundle exec bootsnap precompile -j 1 app/ lib/ && \
    SECRET_KEY_BASE_DUMMY=1 ./bin/rails assets:precompile

FROM base AS python

COPY model/requirements-inference.txt /tmp/requirements-inference.txt

RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y python3 python3-venv python3-pip && \
    rm -rf /var/lib/apt/lists /var/cache/apt/archives && \
    python3 -m venv /opt/counterai/.venv && \
    /opt/counterai/.venv/bin/pip install --no-cache-dir -r /tmp/requirements-inference.txt

FROM base

RUN groupadd --system --gid 1000 rails && \
    useradd rails --uid 1000 --gid 1000 --create-home --shell /bin/bash

COPY --chown=rails:rails --from=build "${BUNDLE_PATH}" "${BUNDLE_PATH}"
COPY --chown=rails:rails --from=build /rails /rails
COPY --chown=rails:rails --from=python /opt/counterai/.venv /opt/counterai/.venv
COPY --chown=rails:rails model/ /model/

USER rails

WORKDIR /rails

ENV PORT="80" \
    CLASSIFIER_PYTHON=/opt/counterai/.venv/bin/python \
    CLASSIFIER_SCRIPT=/model/classify.py \
    CLASSIFIER_CHECKPOINT=/model/artifacts/best_real_fake_20260422_002356_seed42.pt \
    CLASSIFIER_DEVICE=cpu \
    CLASSIFIER_TIMEOUT_SEC=60

ENTRYPOINT ["/rails/bin/docker-entrypoint"]

EXPOSE 80

CMD ["./bin/thrust", "./bin/rails", "server"]
