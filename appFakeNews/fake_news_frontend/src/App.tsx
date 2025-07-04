// src/App.tsx

import { Container, Typography, Box, Paper, Divider } from '@mui/material'; // Importar componentes MUI
import NewsClassifier from './components/NewsClassifier';
import HistoryList from './components/HistoryList';
import ApiStatus from './components/ApiStatus';

function App() {
  return (
    // Usar Container para centralizar e limitar a largura do conteúdo
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Paper elevation={3} sx={{ p: 4, borderRadius: '12px' }}> {/* Adicionar Paper para um card visual */}
        <Typography variant="h3" component="h1" align="center" gutterBottom sx={{ mb: 4 }}>
          Detector de Fake News
        </Typography>

        <Box sx={{ mb: 4 }}>
          <ApiStatus />
        </Box>

        <Divider sx={{ mb: 4 }} /> {/* Separador visual */}

        <Box sx={{ mb: 4 }}>
          <NewsClassifier />
        </Box>

        <Divider sx={{ mb: 4 }} />

        <Box>
          <HistoryList />
        </Box>
      </Paper>
    </Container>
  );
}

export default App;