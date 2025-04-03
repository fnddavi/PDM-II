import React, { useState } from "react";
import { StyleSheet, Text, View, Button } from "react-native";
import axios from "axios";
import { GOOGLE_API_TOKEN } from "@env";

const audio = "gs://cloud-samples-tests/speech/brooklyn.flac";

export default function Exemplo1() {
  const [transcription, setTranscription] = useState("");

  async function transcribeAudio() {
    try {
      const token = GOOGLE_API_TOKEN || undefined;
      console.log("token:", token);
      if (!token) {
        console.error("Erro: Token de autenticação inválido.");
        return;
      }

      const response = await axios.post(
        "https://speech.googleapis.com/v1/speech:recognize",
        {
          config: {
            encoding: "FLAC",
            sampleRateHertz: 16000,
            languageCode: "en-US",
            enableWordTimeOffsets: false,
          },
          audio: {
            uri: audio,
          },
        },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        }
      );

      console.log("Response", response?.data?.results[0]);
      if (response.data && response.data.results) {
        setTranscription(response.data.results[0].alternatives[0].transcript);
      } else {
        console.error("Erro: Nenhuma transcrição retornada pela API.");
      }
    } catch (error: any) {
      console.error("Erro ao transcrever o áudio", error);
    }
  }

  return (
    <View style={styles.container}>
      <Button title="Testar Áudio do Google Cloud" onPress={transcribeAudio} />
      <Text style={styles.transcription}>{transcription}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
  transcription: {
    marginTop: 20,
    textAlign: "center",
    padding: 10,
  },
});
