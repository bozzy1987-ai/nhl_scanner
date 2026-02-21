FROM node:18-alpine as builder

WORKDIR /app

ARG REACT_APP_API_URL=http://localhost:8000
ENV REACT_APP_API_URL=$REACT_APP_API_URL

COPY package*.json ./
RUN npm install

COPY src ./src
COPY public ./public

RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
