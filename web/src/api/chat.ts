import axios from "axios";

export const postChat = async (
  audioBlob: Blob,
  filename = "recording.webm"
): Promise<Blob> => {
  const formData = new FormData();
  formData.append("file", audioBlob, filename);

  const { data } = await axios.post<Blob>("http://localhost:8000/chat", formData, {
    responseType: "blob",
    headers: {
      Accept: "audio/*"
    },
    timeout: 60_000
  });

  return data;
};