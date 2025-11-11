import axios from "axios";

const AI = axios.create({
  baseURL: import.meta.env.VITE_AI_API,
});

export default AI;
