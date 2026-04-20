import { Navigate, Route, BrowserRouter as Router, Routes } from "react-router-dom";
import { AppLayout } from "@/components/layout/app-layout";
import { Analytics } from "@/pages/analytics";
import { Chat } from "@/pages/chat";
import { Dashboard } from "@/pages/dashboard";
import { Fleet } from "@/pages/fleet";
import { Performance } from "@/pages/performance";
import { Settings } from "@/pages/settings";
import { Vision } from "@/pages/vision";

import { Help } from "@/pages/help";

function App() {
  return (
    <Router>
      <AppLayout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/performance" element={<Performance />} />
          <Route path="/vision" element={<Vision />} />
          <Route path="/fleet" element={<Fleet />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/help" element={<Help />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppLayout>
    </Router>
  );
}

export default App;
