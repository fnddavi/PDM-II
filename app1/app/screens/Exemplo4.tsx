import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  TextInput,
  FlatList,
  Alert,
  SafeAreaView,
  TouchableOpacity,
} from "react-native";
import { Audio } from "expo-av";
import { FAB } from "react-native-paper";
import * as FileSystem from "expo-file-system";
import axios from "axios";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";
import { BACKEND_URL } from "@env";

export default function Exemplo4() {
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [title, setTitle] = useState("");
  const [audioList, setAudioList] = useState<
    { title: string; uri: string; transcription?: string }[]
  >([]);

  useEffect(() => {
    loadSavedAudioFiles();
  }, []);

  const loadSavedAudioFiles = async () => {
    try {
      const files = await FileSystem.readDirectoryAsync(
        FileSystem.documentDirectory || ""
      );
      const audioFiles = files
        .filter((file) => file.endsWith(".wav"))
        .map((file) => ({
          title: file.replace(".wav", "").replace(/_/g, " "),
          uri: FileSystem.documentDirectory + file,
          transcription: "",
        }));
      setAudioList(audioFiles);
    } catch (error) {
      Alert.alert("Erro", "Não foi possível carregar os áudios salvos.");
    }
  };

  const startRecording = async () => {
    try {
      await Audio.requestPermissionsAsync();
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const newRecording = new Audio.Recording();
      await newRecording.prepareToRecordAsync(
        Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY
      );
      await newRecording.startAsync();
      setRecording(newRecording);
    } catch (error) {
      Alert.alert("Erro ao iniciar gravação", error.message);
    }
  };

  const stopRecording = async () => {
    if (!recording) return;
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();

    if (!uri) {
      Alert.alert("Erro", "Falha ao obter o arquivo de áudio.");
      return;
    }

    if (!title.trim()) {
      Alert.alert("Erro", "Digite um título para o áudio.");
      return;
    }

    const newUri = `${FileSystem.documentDirectory}${title.replace(
      /\s+/g,
      "_"
    )}.wav`;
    try {
      await FileSystem.moveAsync({ from: uri, to: newUri });
      setAudioList((prev) => [...prev, { title, uri: newUri }]);
      setTitle("");
      setRecording(null);
    } catch (error) {
      Alert.alert("Erro", "Não foi possível salvar o arquivo.");
    }
  };

  const uploadAudio = async (item: { title: string; uri: string }) => {
    try {
      const formData = new FormData();
      formData.append("audio", {
        uri: item.uri,
        name: `${item.title}.wav`,
        type: "audio/wav",
      } as any); // O `as any` é necessário para evitar erro no TypeScript
      
      console.log("BACKEND_URL:", `${BACKEND_URL}/upload`);
      const response = await axios.post(`${BACKEND_URL}/upload`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      console.log("apos", response.data.transcript.results[0].alternatives[0]);

      setAudioList((prev) =>
        prev.map((audio) =>
          audio.uri === item.uri
            ? {
                ...audio,
                transcription:
                  response.data.transcript.results[0].alternatives[0]
                    .transcript,
              }
            : audio
        )
      );

      Alert.alert("Sucesso", "Transcrição concluída com sucesso.");
    } catch (error) {
      Alert.alert("Erro", "Falha ao enviar o áudio para transcrição.");
      console.error("uploadAudio:", error);
    }
  };

  const deleteAudio = async (item: { title: string; uri: string }) => {
    try {
      await FileSystem.deleteAsync(item.uri);
      setAudioList((prev) => prev.filter((audio) => audio.uri !== item.uri));
      Alert.alert("Sucesso", "Áudio excluído com sucesso.");
    } catch (error) {
      Alert.alert("Erro", "Não foi possível excluir o arquivo.");
    }
  };

  return (
    <SafeAreaView style={{ flex: 1, paddingTop: 30 }}>
      <View style={{ flex: 1, padding: 20 }}>
        <TextInput
          placeholder="Título do Áudio"
          value={title}
          onChangeText={setTitle}
          style={{
            borderWidth: 1,
            borderColor: "#ccc",
            padding: 10,
            borderRadius: 5,
            marginBottom: 10,
          }}
        />

        <FlatList
          data={audioList}
          keyExtractor={(item, index) => index.toString()}
          renderItem={({ item }) => (
            <View
              style={{
                padding: 10,
                borderBottomWidth: 1,
              }}
            >
              {/* Linha com título e botões */}
              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <Text style={{ flex: 1, fontWeight: "bold" }}>
                  {item.title}
                </Text>
                <TouchableOpacity style={{marginRight:8}}
                  onPress={() =>
                    Audio.Sound.createAsync({ uri: item.uri }).then(
                      ({ sound }) => sound.playAsync()
                    )
                  }
                >
                  <Icon name="play-circle-outline" size={28} color="blue" />
                </TouchableOpacity>
                <TouchableOpacity onPress={() => uploadAudio(item)} style={{marginRight:8}}>
                  <Icon name="cloud-upload-outline" size={28} color="green" />
                </TouchableOpacity>
                <TouchableOpacity onPress={() => deleteAudio(item)}>
                  <Icon name="delete-outline" size={28} color="red" />
                </TouchableOpacity>
              </View>

              {/* Linha com transcrição (se existir) */}
              {item.transcription ? (
                <Text style={{ marginTop: 5, fontSize: 14, color: "gray" }}>
                  {item.transcription}
                </Text>
              ) : null}
            </View>
          )}
        />

        <FAB
          style={{
            position: "absolute",
            bottom: 20,
            right: 20,
            backgroundColor: !title ? "#888" : recording ? "orange" : "green",
          }}
          icon={recording ? "stop" : "microphone"}
          onPress={recording ? stopRecording : startRecording}
          disabled={!title.trim()}
        />
      </View>
    </SafeAreaView>
  );
}
