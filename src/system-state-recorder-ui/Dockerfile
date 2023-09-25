FROM node:latest 
WORKDIR /dashbaord
COPY package.json ./
COPY package-lock.json ./
COPY ./ ./
RUN npm install 
CMD ["npm", "run", "start"]
