# Two stages. Node builds the bundle; nginx serves it. The runtime image
# carries no node_modules and no toolchain, which takes it from roughly
# 400 MB to under 60 MB and removes every build dependency from the thing
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

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD wget -qO- http://localhost/ >/dev/null || exit 1
