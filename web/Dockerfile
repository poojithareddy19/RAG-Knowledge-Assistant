# Two stages. Node builds the bundle; nginx serves it. The runtime image
# carries no node_modules and no toolchain, which takes it from roughly
# 400 MB to about 75 MB (74 MB measured, most of it Plotly) and removes every
# build dependency from the thing
# facing the internet.

FROM node:22-alpine AS build

WORKDIR /app

# Manifest first so the dependency layer caches independently of source edits.
COPY package.json package-lock.json* ./
RUN npm install

COPY . .
RUN npm run build


FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# 127.0.0.1, not localhost. In the container localhost resolves to ::1 first,
# and nginx here listens on IPv4 only, so the check got "connection refused"
# from a server that was answering every real request. The first run of the
# production compose reported web as unhealthy while it served the front end
# and proxied the API without a fault. A healthcheck that disagrees with the
# service it checks is worse than none: it teaches you to ignore red.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
